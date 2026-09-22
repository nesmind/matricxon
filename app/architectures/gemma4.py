import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.gemma4_layers import Gemma4DecoderLayer
from app.architectures.layers import RMSNorm
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Gemma4Architecture(ModelArchitecture):
    """The dense (non-MoE) `gemma4` decoder - confirmed via a real

    `google/gemma-4-12b-it` GGUF pull (7GB, Q4_0) that this specific
    checkpoint carries no expert/routing metadata at all, unlike the MoE
    variant the real HF `transformers.models.gemma4` package also supports
    (`Gemma4TextExperts`/`Gemma4TextRouter`) - this class only ports the
    code path that checkpoint's own config actually activates (also
    confirmed inactive for this checkpoint: KV-sharing across layers,
    the per-layer-input gating branch).

    Genuinely different from `Mistral3TextArchitecture`/`LlamaArchitecture`
    in ways real tensor inspection surfaced, not assumed from general Gemma
    familiarity - see `gemma4_layers.py`'s own docstrings for the per-layer
    attention shape/behavior differences. At the whole-model level: two
    separate RoPE configurations (local/sliding layers use `freq_base_swa`
    at `head_dim`=256; global layers use `freq_base`=1e6 at `head_dim`=512,
    corrected by a real `rope_freqs.weight` tensor - see
    `_materialize_weights`), which layers are local vs. global comes from
    the real per-layer `attention.sliding_window_pattern` metadata array
    (not a hardcoded period), scaled input embeddings (`* sqrt(n_embd)`,
    the classic Gemma-family convention), and final-logit soft-capping
    (`tanh(logits / cap) * cap`, real value 30.0 on this checkpoint).
    """

    NAME = "gemma4"

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> None:
        super().__init__()
        # Unlike mistral3/llama, this architecture needs no `unpermute_rope_rows` at all (real,
        # confirmed: no such call anywhere in this file) - so q_proj/k_proj go quantized-native
        # too when this is on, alongside v_proj (when not None)/o_proj/gate_proj/up_proj/
        # down_proj. No separate lm_head - tied to token_embd, same as mistral3.
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        self.vocab_size = len(metadata.require("tokenizer.ggml.tokens"))
        self.final_logit_softcapping = metadata.get_f32(arch("final_logit_softcapping"))
        self.sliding_window = metadata.get_u32(arch("attention.sliding_window"))

        head_count_kv = metadata.require(arch("attention.head_count_kv"))
        self._n_head_kv: list[int] = (
            list(head_count_kv)
            if isinstance(head_count_kv, list)
            else [head_count_kv] * self.n_layer
        )
        sliding_pattern = metadata.get_array(arch("attention.sliding_window_pattern"))
        self._is_sliding: list[bool] = (
            list(sliding_pattern) if sliding_pattern is not None else [False] * self.n_layer
        )

        self._head_dim_global = metadata.get_u32(arch("attention.key_length"))
        self._head_dim_local = metadata.get_u32(
            arch("attention.key_length_swa"), self._head_dim_global
        )

        # Plain floats (not tensors), so unaffected by _construct_without_init's meta-device
        # trick - kept so _rebuild_derived_buffers can recreate the real rope modules
        # afterward. See Mistral3TextArchitecture's identical _rope_kwargs for the reasoning.
        self._rope_kwargs_local = dict(
            head_dim=self._head_dim_local, rope_theta=metadata.get_f32(arch("rope.freq_base_swa"))
        )
        self._rope_kwargs_global = dict(
            head_dim=self._head_dim_global, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self.rope_local = RotaryEmbedding(**self._rope_kwargs_local)
        self.rope_global = RotaryEmbedding(**self._rope_kwargs_global)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self._n_head_kv[i],
                    self._head_dim_local if self._is_sliding[i] else self._head_dim_global,
                    self.ffn_len,
                    self.rms_eps,
                    use_v_from_k=not self._is_sliding[i],
                    sliding_window=self.sliding_window if self._is_sliding[i] else None,
                    dtype=dtype,
                )
                for i in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        # No separate "output.weight" tensor in GGUF for this model - lm_head
        # is tied to token_embd (confirmed: no such tensor in the real file).

    def _rebuild_derived_buffers(self) -> None:
        """Both ropes' inv_freq are real computed values, not GGUF tensors from_gguf ever

        `.copy_()`'s in directly - see ModelArchitecture._rebuild_derived_buffers's own
        docstring for why they have to be recreated for real after _construct_without_init.
        `rope_global`'s inv_freq gets a further real correction from the actual
        `rope_freqs.weight` tensor once that's available - see `_materialize_weights`.
        """
        self.rope_local = RotaryEmbedding(**self._rope_kwargs_local)
        self.rope_global = RotaryEmbedding(**self._rope_kwargs_global)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        return [
            (
                self._n_head_kv[i],
                self._head_dim_local if self._is_sliding[i] else self._head_dim_global,
            )
            for i in range(self.n_layer)
        ]

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "Gemma4Architecture":
        model = cls._construct_without_init(
            loader.metadata, dtype=dtype, enable_quantized_native=enable_quantized_native
        )
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        if loader.has_tensor("rope_freqs.weight"):
            # Real llama.cpp `freq_factors` convention (confirmed on this checkpoint, not
            # assumed): divides the formulaic inv_freq per-dimension. Only ever present for
            # the global rope (shape matches head_dim_global/2 on the real file).
            rope_freqs = loader.load_tensor("rope_freqs.weight")
            self.rope_global.inv_freq.copy_(self.rope_global.inv_freq / rope_freqs)
        logger.debug(
            "materializing: token_embd + output_norm in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            enabled = self._enable_quantized_native
            layer.self_attn.q_proj = self._load_projection(
                loader, prefix + "attn_q.weight", layer.self_attn.q_proj, self._dtype, enabled
            )
            layer.self_attn.q_norm.weight.copy_(loader.load_tensor(prefix + "attn_q_norm.weight"))
            layer.self_attn.k_proj = self._load_projection(
                loader, prefix + "attn_k.weight", layer.self_attn.k_proj, self._dtype, enabled
            )
            layer.self_attn.k_norm.weight.copy_(loader.load_tensor(prefix + "attn_k_norm.weight"))
            if layer.self_attn.v_proj is not None:
                layer.self_attn.v_proj = self._load_projection(
                    loader, prefix + "attn_v.weight", layer.self_attn.v_proj, self._dtype, enabled
                )
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, self._dtype, enabled
            )
            layer.post_attention_layernorm.weight.copy_(
                loader.load_tensor(prefix + "post_attention_norm.weight")
            )
            layer.pre_feedforward_layernorm.weight.copy_(
                loader.load_tensor(prefix + "ffn_norm.weight")
            )
            layer.mlp.gate_proj = self._load_projection(
                loader, prefix + "ffn_gate.weight", layer.mlp.gate_proj, self._dtype, enabled
            )
            layer.mlp.up_proj = self._load_projection(
                loader, prefix + "ffn_up.weight", layer.mlp.up_proj, self._dtype, enabled
            )
            layer.mlp.down_proj = self._load_projection(
                loader, prefix + "ffn_down.weight", layer.mlp.down_proj, self._dtype, enabled
            )
            layer.post_feedforward_layernorm.weight.copy_(
                loader.load_tensor(prefix + "post_ffw_norm.weight")
            )
            layer.layer_scalar.data.copy_(loader.load_tensor(prefix + "layer_output_scale.weight"))
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
        del image_embeddings  # real vision support is LlamaArchitecture-only - see base.py
        _, seq_len = input_ids.shape
        logger.debug("forward start: seq_len=%d", seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        stage_started = time.monotonic()
        cos_local, sin_local = self.rope_local(position_ids)
        cos_global, sin_global = self.rope_global(position_ids)
        logger.debug("rope (local+global): %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.token_embd(input_ids) * (self.n_embd**0.5)
        logger.debug("embedding lookup: %.1fms", (time.monotonic() - stage_started) * 1000)

        for i, layer in enumerate(self.layers):
            cos, sin = (cos_local, sin_local) if self._is_sliding[i] else (cos_global, sin_global)
            layer_started = time.monotonic()
            x = layer(x, cos, sin, kv_cache, i)
            logger.debug(
                "layer %d/%d: %.1fms (seq_len=%d, sliding=%s)",
                i + 1,
                self.n_layer,
                (time.monotonic() - layer_started) * 1000,
                seq_len,
                self._is_sliding[i],
            )
            if stop_check is not None and stop_check():
                logger.info(
                    "generation stop requested - cancelling after layer %d/%d", i + 1, self.n_layer
                )
                raise GenerationCancelledError(f"stopped after layer {i + 1}/{self.n_layer}")

        stage_started = time.monotonic()
        x = self.output_norm(x)

        logits = F.linear(x, self.token_embd.weight)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        logger.debug(
            "output_norm + lm_head: %.1fms (logits shape=%s)",
            (time.monotonic() - stage_started) * 1000,
            tuple(logits.shape),
        )
        return logits
