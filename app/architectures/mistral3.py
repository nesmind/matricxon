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
from app.architectures.packed_weights import PACKED_EMBEDDINGS
from app.architectures.rope import YarnRotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.models.load_dtype import NON_LAYER_GROUP
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Mistral3TextArchitecture(ModelArchitecture):
    """The text-decoder half of GGUF's "mistral3" arch.

    Vision tower (`v.*`) and multimodal projector (`mm.*`) tensors are never requested from the
    loader, so never read or
    dequantized: the mechanism behind matricxon's v1 "accept-and-ignore images" decision. Building
    blocks: mistral3_layers.py.
    """

    NAME = "mistral3"
    supports_context_shift = True
    SUPPORTS_GPU = True
    SUPPORTS_BATCHED_DECODE = True

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        layer_dtypes: dict[str, torch.dtype] | None = None,
        enable_quantized_native: bool = False,
    ) -> None:
        super().__init__()
        layer_dtypes = layer_dtypes or {}
        # See Settings.enable_quantized_native_compute: a permanent user-selectable choice, not a
        # rollout flag. When on, all 7
        # per-layer projections go quantized-native (`q_proj`/`k_proj` with `unpermute_rope_rows` on
        # their packed rows;
        # `PackedWeightLoading._load_projection`), and so does the tied token_embd/lm_head table
        # (`QuantizedEmbedding`).
        self._enable_quantized_native = enable_quantized_native
        self.n_embd = metadata.get_u32(metadata.arch_key("embedding_length"))
        self.n_head = metadata.get_u32(metadata.arch_key("attention.head_count"))
        # Falls back to n_head when absent (no GQA); see LlamaArchitecture.
        self.n_head_kv = metadata.get_u32(metadata.arch_key("attention.head_count_kv"), self.n_head)
        self.head_dim = metadata.get_u32(metadata.arch_key("attention.key_length"))
        self.n_layer = metadata.get_u32(metadata.arch_key("block_count"))
        self.ffn_len = metadata.get_u32(metadata.arch_key("feed_forward_length"))
        self.rms_eps = metadata.get_f32(metadata.arch_key("attention.layer_norm_rms_epsilon"))
        self.vocab_size = metadata.get_u32(metadata.arch_key("vocab_size"))

        # beta_fast/beta_slow: GGUFs write `yarn_beta_fast`/`yarn_beta_slow` (seen in a Ministral-3
        # pull), not the unprefixed
        # names of HF's rope_scaling dict, as this first assumed. Falls back to the unprefixed name
        # in case another conversion
        # omits the prefix, then to HF's `_compute_yarn_parameters` defaults (32.0/1.0), not
        # crashing on `None * int`.
        beta_fast = metadata.get_f32(metadata.arch_key("rope.scaling.yarn_beta_fast"))
        if beta_fast is None:
            beta_fast = metadata.get_f32(metadata.arch_key("rope.scaling.beta_fast"), 32.0)
        beta_slow = metadata.get_f32(metadata.arch_key("rope.scaling.yarn_beta_slow"))
        if beta_slow is None:
            beta_slow = metadata.get_f32(metadata.arch_key("rope.scaling.beta_slow"), 1.0)

        # Kept (plain floats/ints, unaffected by _construct_without_init's meta-device trick) so
        # _rebuild_derived_buffers can
        # recreate self.rope without re-parsing metadata or duplicating the beta_fast/beta_slow
        # fallback.
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

        # Per-layer/per-group dtype overrides (ModelManager._load's mixed-precision branch,
        # app.models.load_dtype.plan_layer_dtypes); empty `layer_dtypes` means every group uses the
        # uniform `dtype`.
        # `self._layer_dtypes` is kept (not re-derived from each layer's parameter dtype) so
        # `_forward_impl` has an O(1) lookup
        # for its required inter-layer dtype casts (see its docstring).
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
                    self.rms_eps,
                    mlp=SwiGLUMLP(self.n_embd, self.ffn_len, dtype=self._layer_dtypes[i]),
                    dtype=self._layer_dtypes[i],
                )
                for i in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=embed_dtype)
        # No "output.weight" in the GGUF: lm_head is tied to token_embd (HF config:
        # tie_word_embeddings=true).

    def _rebuild_derived_buffers(self) -> None:
        """self.rope.inv_freq is a computed value, not a GGUF tensor from_gguf `.copy_`'s in;
        recreated after
        _construct_without_init (ModelArchitecture._rebuild_derived_buffers).
        """
        self.rope = YarnRotaryEmbedding(**self._rope_kwargs)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        """The same `(n_head_kv, head_dim)` pair per layer (`KVCache` says why it's per layer:
        Gemma4's per-layer-type head dims need it; mistral3's don't)."""
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
        """`dtype` sizes the module's parameters (float32 by default, the M3 oracle's validated
        numerics). It is independent of `loader`'s dequant dtype (`.copy_` casts) and is an
        argument, not a `torch.set_default_dtype` context, which would race if another model's
        `ModelWorker` thread builds default-dtype tensors mid-forward.

        `layer_dtypes` (app.models.load_dtype.plan_layer_dtypes) lets ModelManager give individual
        decoder layers, and the tied embedding/lm_head (key NON_LAYER_GROUP), a dtype other than
        `dtype`, so a model that doesn't fully fit float32 runs as many layers at float32 speed as
        fit instead of all bf16 (see ModelManager._load).

        `enable_quantized_native`: see Settings.enable_quantized_native_compute.
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
        if isinstance(self.token_embd, PACKED_EMBEDDINGS):
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
        """The `x.to(...)` casts before each layer and before the tied lm_head matmul are needed:
        with uniform dtypes (`layer_dtypes` unset) they are no-ops, but with mixed per-layer dtypes
        `RMSNorm.forward` casts its output to its input's dtype, so the residual stream can carry
        the previous layer's dtype into a layer built with another. Elementwise ops promote, but the
        next nn.Linear matmul raises RuntimeError on a mismatch (see ModelManager._load).
        apply_rotary_pos_emb guards cos/sin vs q/k likewise."""
        del image_embeddings
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
