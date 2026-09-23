import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.layers import RMSNorm
from app.architectures.mistral3_layers import (  # noqa: F401 - re-exported for existing importers
    GroupedQueryAttention,
    Mistral3DecoderLayer,
    SwiGLUMLP,
)
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.rope import YarnRotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.models.load_dtype import NON_LAYER_GROUP
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Mistral3TextArchitecture(ModelArchitecture):
    """The text-decoder half of GGUF's "mistral3" arch.

    Vision tower (`v.*`) and multimodal projector (`mm.*`) tensors are never
    requested from the loader below, so they're never read or dequantized -
    this is the mechanism behind matricxon's v1 "accept-and-ignore images"
    decision, not a separate flag. See mistral3_layers.py for the decoder
    layer/attention/MLP building blocks this assembles.
    """

    NAME = "mistral3"

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        layer_dtypes: dict[str, torch.dtype] | None = None,
        enable_quantized_native: bool = False,
    ) -> None:
        super().__init__()
        layer_dtypes = layer_dtypes or {}
        # See Settings.enable_quantized_native_compute's own docstring - a real, permanent,
        # user-selectable choice, not a temporary rollout flag. When on, all 7 per-layer
        # projections go quantized-native - `q_proj`/`k_proj` with `unpermute_rope_rows` applied
        # to their packed rows (see `PackedWeightLoading._load_projection`) - and so does the
        # tied token_embd/lm_head table (`QuantizedEmbedding`).
        self._enable_quantized_native = enable_quantized_native
        self.n_embd = metadata.get_u32(metadata.arch_key("embedding_length"))
        self.n_head = metadata.get_u32(metadata.arch_key("attention.head_count"))
        self.n_head_kv = metadata.get_u32(metadata.arch_key("attention.head_count_kv"))
        self.head_dim = metadata.get_u32(metadata.arch_key("attention.key_length"))
        self.n_layer = metadata.get_u32(metadata.arch_key("block_count"))
        self.ffn_len = metadata.get_u32(metadata.arch_key("feed_forward_length"))
        self.rms_eps = metadata.get_f32(metadata.arch_key("attention.layer_norm_rms_epsilon"))
        self.vocab_size = metadata.get_u32(metadata.arch_key("vocab_size"))

        # beta_fast/beta_slow: real GGUFs write these as `yarn_beta_fast`/`yarn_beta_slow`
        # (confirmed against a real Ministral-3 pull) - not the unprefixed `beta_fast`/
        # `beta_slow` HF's own rope_scaling config dict uses, which this originally (wrongly)
        # assumed carried over 1:1 into GGUF. Falls back to the unprefixed name in case some
        # other real conversion omits the prefix, then to HF's own _compute_yarn_parameters
        # defaults (32.0/1.0) if neither is present, rather than crashing on `None * int`.
        beta_fast = metadata.get_f32(metadata.arch_key("rope.scaling.yarn_beta_fast"))
        if beta_fast is None:
            beta_fast = metadata.get_f32(metadata.arch_key("rope.scaling.beta_fast"), 32.0)
        beta_slow = metadata.get_f32(metadata.arch_key("rope.scaling.yarn_beta_slow"))
        if beta_slow is None:
            beta_slow = metadata.get_f32(metadata.arch_key("rope.scaling.beta_slow"), 1.0)

        # Kept around (plain floats/ints, not tensors - unaffected by _construct_without_init's
        # meta-device trick) so _rebuild_derived_buffers can recreate self.rope for real afterward,
        # without re-parsing metadata or duplicating the beta_fast/beta_slow fallback logic above.
        self._rope_kwargs = dict(
            head_dim=self.head_dim,
            rope_theta=metadata.get_f32(metadata.arch_key("rope.freq_base")),
            factor=metadata.get_f32(metadata.arch_key("rope.scaling.factor")),
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            original_context_length=metadata.get_u32(
                metadata.arch_key("rope.scaling.original_context_length")
            ),
            mscale=metadata.get_f32(metadata.arch_key("rope.scaling.mscale")),
            mscale_all_dim=metadata.get_f32(metadata.arch_key("rope.scaling.mscale_all_dim")),
        )
        self.rope = YarnRotaryEmbedding(**self._rope_kwargs)

        # Per-layer/per-group dtype overrides (see ModelManager._load's mixed-precision branch,
        # app.models.load_dtype.plan_layer_dtypes) - `layer_dtypes` empty (the default) means every
        # group falls back to the uniform `dtype`, i.e. today's exact single-dtype behavior.
        # `self._layer_dtypes` is kept (not re-derived from each layer's own live parameter dtype)
        # so `_forward_impl` has an O(1) list lookup for its required inter-layer dtype casts (see
        # that method's own docstring for why those casts are necessary, not optional).
        embed_dtype = layer_dtypes.get(NON_LAYER_GROUP, dtype)
        self._layer_dtypes = [layer_dtypes.get(f"blk.{i}", dtype) for i in range(self.n_layer)]

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=embed_dtype)
        self.layers = nn.ModuleList(
            [
                Mistral3DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.ffn_len,
                    self.rms_eps,
                    dtype=self._layer_dtypes[i],
                )
                for i in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=embed_dtype)
        # No separate "output.weight" tensor in GGUF for this model - lm_head
        # is tied to token_embd (confirmed via HF config: tie_word_embeddings=true).

    def _rebuild_derived_buffers(self) -> None:
        """self.rope.inv_freq is a real computed value, not a GGUF tensor from_gguf ever
        `.copy_()`'s in - see ModelArchitecture._rebuild_derived_buffers's own docstring for why
        this has to be recreated for real after _construct_without_init."""
        self.rope = YarnRotaryEmbedding(**self._rope_kwargs)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        """Same `(n_head_kv, head_dim)` pair repeated per layer - see

        `KVCache`'s own docstring for why this is a per-layer list at all
        (Gemma4's real per-layer-type head dims need it; mistral3's don't).
        """
        return [(self.n_head_kv, self.head_dim)] * self.n_layer

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        layer_dtypes: dict[str, torch.dtype] | None = None,
        enable_quantized_native: bool = False,
    ) -> "Mistral3TextArchitecture":
        """`dtype` sizes the module's own real parameters (float32 by
        default, matching the M3 oracle's validated numerics). It's
        independent of `loader`'s own dequant dtype - `.copy_()` below casts
        regardless - and is a plain constructor argument rather than a
        `torch.set_default_dtype()` context, which would be a data race if
        another model's `ModelWorker` thread is concurrently building
        default-dtype-dependent tensors of its own during a forward pass.

        `layer_dtypes` (see app.models.load_dtype.plan_layer_dtypes) lets ModelManager give
        individual decoder layers - and the tied embedding/lm_head, keyed by
        app.models.load_dtype.NON_LAYER_GROUP - a dtype different from the uniform `dtype`
        fallback, so a model that doesn't fully fit float32 can still run as many of its layers at
        float32 speed as actually fit, instead of the whole model falling back to bf16 (see
        ModelManager._load's own docstring on why this exists and when it's actually used - not
        every load computes or needs this).

        `enable_quantized_native` - see Settings.enable_quantized_native_compute's own docstring.
        """
        model = cls._construct_without_init(
            loader.metadata,
            dtype=dtype,
            layer_dtypes=layer_dtypes,
            enable_quantized_native=enable_quantized_native,
        )
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        enabled = self._enable_quantized_native
        embed_dtype = self.token_embd.weight.dtype
        self.token_embd = self._load_token_embedding(loader, self.token_embd, embed_dtype, enabled)
        if isinstance(self.token_embd, QuantizedEmbedding):
            self.lm_head = self.token_embd.as_linear()
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        logger.debug(
            "materializing: token_embd + output_norm in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            layer.post_attention_layernorm.weight.copy_(
                loader.load_tensor(prefix + "ffn_norm.weight")
            )
            dtype = self._layer_dtypes[i]
            layer.self_attn.q_proj = self._load_projection(
                loader,
                prefix + "attn_q.weight",
                layer.self_attn.q_proj,
                dtype,
                enabled,
                rope_heads=self.n_head,
            )
            layer.self_attn.k_proj = self._load_projection(
                loader,
                prefix + "attn_k.weight",
                layer.self_attn.k_proj,
                dtype,
                enabled,
                rope_heads=self.n_head_kv,
            )
            layer.self_attn.v_proj = self._load_projection(
                loader, prefix + "attn_v.weight", layer.self_attn.v_proj, dtype, enabled
            )
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, dtype, enabled
            )
            layer.mlp.gate_proj = self._load_projection(
                loader, prefix + "ffn_gate.weight", layer.mlp.gate_proj, dtype, enabled
            )
            layer.mlp.up_proj = self._load_projection(
                loader, prefix + "ffn_up.weight", layer.mlp.up_proj, dtype, enabled
            )
            layer.mlp.down_proj = self._load_projection(
                loader, prefix + "ffn_down.weight", layer.mlp.down_proj, dtype, enabled
            )
            logger.debug(
                "loading weights: layer %d/%d (not computing yet) in %.1fs",
                i + 1,
                self.n_layer,
                time.monotonic() - layer_started,
            )
            if stop_check is not None and stop_check():
                raise GenerationCancelledError(f"stopped loading weights at layer {i + 1}")

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Explicit `x.to(...)` casts before each layer and before the final tied lm_head matmul
        are required, not defensive belt-and-suspenders: with uniform-dtype loading (`layer_dtypes`
        unset) every one of these casts is a no-op (source and target dtype already match), but
        under mixed per-layer dtypes, `RMSNorm.forward` (app/architectures/layers.py) casts its
        output back to *its input's* dtype, not necessarily its own weight's dtype - so the
        residual stream `x` can silently carry the previous layer's dtype into a layer built with a
        different one. Elementwise ops (RMSNorm's own internal multiply) implicitly promote mixed
        dtypes without erroring, but the matmul inside the next layer's nn.Linear does not - it
        raises a real RuntimeError on a dtype mismatch, confirmed by reading both code paths (see
        ModelManager._load's own docstring for the full reasoning behind mixed per-layer dtypes at
        all). apply_rotary_pos_emb already defends against the equivalent cos/sin-vs-q/k case the
        same way (casts to q.dtype) - these are that same pattern, applied at layer boundaries.
        """
        del image_embeddings  # real vision support is LlamaArchitecture-only - see base.py
        _, seq_len = input_ids.shape
        logger.debug("forward start: seq_len=%d", seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        stage_started = time.monotonic()
        cos, sin = self.rope(position_ids)
        logger.debug("rope: %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.token_embd(input_ids)
        logger.debug("embedding lookup: %.1fms", (time.monotonic() - stage_started) * 1000)

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x.to(self._layer_dtypes[i]), cos, sin, kv_cache, i)
            logger.debug(
                "layer %d/%d: %.1fms (seq_len=%d)",
                i + 1,
                self.n_layer,
                (time.monotonic() - layer_started) * 1000,
                seq_len,
            )
            if stop_check is not None and stop_check():
                logger.info(
                    "generation stop requested - cancelling after layer %d/%d", i + 1, self.n_layer
                )
                raise GenerationCancelledError(f"stopped after layer {i + 1}/{self.n_layer}")

        stage_started = time.monotonic()
        if self._last_logits_only:
            x = x[:, -1:, :]
        x = self.output_norm(x.to(self.output_norm.weight.dtype))
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None:
            logits = lm_head(x)
        else:
            logits = F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        logger.debug(
            "output_norm + lm_head: %.1fms (logits shape=%s)",
            (time.monotonic() - stage_started) * 1000,
            tuple(logits.shape),
        )
        return logits
