import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.gemma4_layers import Gemma4DecoderLayer
from app.architectures.gemma4_moe import Gemma4MoEBlock, detect_moe, materialize_moe
from app.architectures.gemma4_ple import Gemma4PerLayerEmbedding
from app.architectures.layers import RMSNorm
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Gemma4Architecture(ModelArchitecture):
    """The dense (non-MoE) `gemma4` decoder - first built against a real
    `google/gemma-4-12b-it` GGUF pull (7GB, Q4_0, no expert/routing metadata, no active
    KV-sharing/PLE), later extended (2026-09-29) against a real `gemma-4-E2B-it` GGUF that DOES
    activate both real Per-Layer Embeddings (see `gemma4_ple.py`) and cross-layer KV reuse (see
    `Gemma4Attention`'s own docstring) - this class ports whatever a given real checkpoint's own
    metadata actually activates, never assumed uniform across checkpoints. MoE (real HF
    `Gemma4TextExperts`/`Gemma4TextRouter`) is still out of scope - no real GGUF needing it has
    been checked against yet.

    Genuinely different from `Mistral3TextArchitecture`/`LlamaArchitecture` in ways real tensor
    inspection surfaced, not assumed from general Gemma familiarity - see `gemma4_layers.py`'s
    own docstrings for the per-layer attention shape/behavior differences. At the whole-model
    level: two separate RoPE configurations (local/sliding layers use `freq_base_swa` at
    `head_dim`=256; global layers use `freq_base`=1e6 at `head_dim`=512, corrected by a real
    `rope_freqs.weight` tensor - see `_materialize_weights`), which layers are local vs. global
    comes from the real per-layer `attention.sliding_window_pattern` metadata array (not a
    hardcoded period), scaled input embeddings (`* sqrt(n_embd)`, the classic Gemma-family
    convention), and final-logit soft-capping (`tanh(logits / cap) * cap`, real value 30.0 on
    the original 12B checkpoint).
    """

    NAME = "gemma4"
    SUPPORTS_MOE = True  # real Gemma4TextExperts/Gemma4TextRouter - see gemma4_moe.py

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
        has_attn_v: list[bool] | None = None,
        has_moe: list[bool] | None = None,
    ) -> None:
        super().__init__()
        # No unpermute_rope_rows anywhere here (unlike mistral3/llama) - q_proj/k_proj go
        # quantized-native too when this is on. No separate lm_head - tied to token_embd.
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        self.vocab_size = len(metadata.require("tokenizer.ggml.tokens"))
        self.final_logit_softcapping = metadata.get_f32(arch("final_logit_softcapping"))
        self.sliding_window = metadata.get_u32(arch("attention.sliding_window"))

        # A real per-layer array on gemma-4-E2B-it (the shared-kv layers double their real ffn
        # width there - llama.cpp's own `use_double_wide_mlp`), a uniform scalar on the 12B
        # checkpoint - `isinstance` below covers both without assuming either.
        ffn_len = metadata.require(arch("feed_forward_length"))
        self.ffn_len_per_layer: list[int] = (
            list(ffn_len) if isinstance(ffn_len, list) else [ffn_len] * self.n_layer
        )

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

        # Real cross-layer KV reuse (HF `num_kv_shared_layers`/llama.cpp `n_layer_kv_from_start`)
        # - see Gemma4Attention's own docstring for what this means at forward time.
        # `_kv_provider_idx` is llama.cpp's own fixed formula (`src/llama-model.cpp`'s `reuse`
        # callback for LLM_ARCH_GEMMA3N/GEMMA4), not a "last occurrence of this type" search.
        n_kv_shared_layers = metadata.get_u32(arch("attention.shared_kv_layers"), 0) or 0
        self._n_layer_kv_from_start = self.n_layer - n_kv_shared_layers
        self._has_own_kv: list[bool] = [
            i < self._n_layer_kv_from_start for i in range(self.n_layer)
        ]
        self._kv_provider_idx: dict[bool, int] = (
            {True: self._n_layer_kv_from_start - 2, False: self._n_layer_kv_from_start - 1}
            if n_kv_shared_layers > 0
            else {}
        )
        self._kv_provider_layers = set(self._kv_provider_idx.values())
        has_attn_v = has_attn_v if has_attn_v is not None else [True] * self.n_layer

        # Real Mixture-of-Experts (see gemma4_moe.py's own docstring for why this can't reuse
        # GraniteMoeFFN/llama_moe unchanged) - `expert_count`/`expert_used_count` are whole-model
        # metadata (real HF applies one shared `enable_moe_block` to every layer), but which
        # layers actually carry expert tensors is still checked per real tensor presence
        # (`has_moe`, from `from_gguf`) - the same real-fact-over-assumption discipline
        # `has_attn_v` above already uses, in case a real checkpoint ever mixes MoE/dense layers.
        num_experts, num_experts_per_tok, moe_ffn_len = detect_moe(metadata, metadata.architecture)
        self.is_moe = num_experts is not None
        has_moe = has_moe if has_moe is not None else [self.is_moe] * self.n_layer

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

        # Per-Layer Embeddings - real, active on gemma-4-E2B-it, 0 on the 12B checkpoint (hence
        # the guard, not an unconditional build). See gemma4_ple.py's own docstring.
        self.per_layer_dim = metadata.get_u32(arch("embedding_length_per_layer_input"), 0) or 0
        if self.per_layer_dim:
            self.per_layer_embedding = Gemma4PerLayerEmbedding(
                self.vocab_size,
                self.n_layer,
                self.per_layer_dim,
                self.n_embd,
                self.rms_eps,
                dtype=dtype,
            )

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self._n_head_kv[i],
                    self._head_dim_local if self._is_sliding[i] else self._head_dim_global,
                    self.ffn_len_per_layer[i],
                    self.rms_eps,
                    has_own_kv=self._has_own_kv[i],
                    use_v_from_k=not has_attn_v[i],
                    sliding_window=self.sliding_window if self._is_sliding[i] else None,
                    per_layer_dim=self.per_layer_dim,
                    moe=(
                        Gemma4MoEBlock(
                            self.n_embd,
                            moe_ffn_len,
                            num_experts,
                            num_experts_per_tok,
                            self.rms_eps,
                            dtype=dtype,
                        )
                        if self.is_moe and has_moe[i]
                        else None
                    ),
                    dtype=dtype,
                )
                for i in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        # No separate "output.weight" tensor in GGUF for this model - lm_head
        # is tied to token_embd (confirmed: no such tensor in the real file).

    def _rebuild_derived_buffers(self) -> None:
        """Both ropes' inv_freq are computed, not GGUF tensors from_gguf `.copy_()`'s in - see
        ModelArchitecture._rebuild_derived_buffers for why. `rope_global`'s gets a further real
        correction from `rope_freqs.weight` once available - see `_materialize_weights`."""
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
        # Real per-layer attn_v presence (see Gemma4Attention's docstring) - checked before
        # __init__ builds any module, since `use_v_from_k` decides whether a layer's v_proj
        # submodule exists at all. Cheap: has_tensor indexes tensor_infos, no byte read.
        arch = loader.metadata.arch_key
        n_layer = loader.metadata.get_u32(arch("block_count"))
        has_attn_v = [loader.has_tensor(f"blk.{i}.attn_v.weight") for i in range(n_layer)]
        # Real per-layer expert-tensor presence - same reasoning as has_attn_v above (see
        # __init__'s own comment on why MoE is still checked per layer, not assumed uniform).
        has_moe = [loader.has_tensor(f"blk.{i}.ffn_gate_inp.weight") for i in range(n_layer)]
        model = cls._construct_without_init(
            loader.metadata,
            dtype=dtype,
            enable_quantized_native=enable_quantized_native,
            has_attn_v=has_attn_v,
            has_moe=has_moe,
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
        if self.per_layer_dim:
            self.per_layer_embedding.embed_tokens_per_layer.weight.copy_(
                loader.load_tensor("per_layer_token_embd.weight")
            )
            self.per_layer_embedding.per_layer_model_projection.weight.copy_(
                loader.load_tensor("per_layer_model_proj.weight")
            )
            self.per_layer_embedding.per_layer_projection_norm.weight.copy_(
                loader.load_tensor("per_layer_proj_norm.weight")
            )
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
            if self._has_own_kv[i]:
                layer.self_attn.k_proj = self._load_projection(
                    loader, prefix + "attn_k.weight", layer.self_attn.k_proj, self._dtype, enabled
                )
                layer.self_attn.k_norm.weight.copy_(
                    loader.load_tensor(prefix + "attn_k_norm.weight")
                )
                if layer.self_attn.v_proj is not None:
                    layer.self_attn.v_proj = self._load_projection(
                        loader,
                        prefix + "attn_v.weight",
                        layer.self_attn.v_proj,
                        self._dtype,
                        enabled,
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
            if layer.moe is not None:
                materialize_moe(layer.moe, loader, prefix)
            layer.post_feedforward_layernorm.weight.copy_(
                loader.load_tensor(prefix + "post_ffw_norm.weight")
            )
            layer.layer_scalar.data.copy_(loader.load_tensor(prefix + "layer_output_scale.weight"))
            if self.per_layer_dim:
                layer.per_layer_input_gate.weight.copy_(
                    loader.load_tensor(prefix + "inp_gate.weight")
                )
                layer.per_layer_projection.weight.copy_(loader.load_tensor(prefix + "proj.weight"))
                layer.post_per_layer_input_norm.weight.copy_(
                    loader.load_tensor(prefix + "post_norm.weight")
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

        # PLE: computed once, sliced per layer below (see gemma4_ple.py). None if per_layer_dim==0.
        per_layer_inputs = self.per_layer_embedding(input_ids, x) if self.per_layer_dim else None

        # Cross-layer KV reuse: populated by each provider layer as it runs (see
        # Gemma4Attention's docstring), read by later shared layers - never persisted across calls.
        shared_kv: dict[bool, tuple[torch.Tensor, torch.Tensor]] = {}

        for i, layer in enumerate(self.layers):
            cos, sin = (cos_local, sin_local) if self._is_sliding[i] else (cos_global, sin_global)
            layer_started = time.monotonic()
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            layer_shared_kv = None if self._has_own_kv[i] else shared_kv[self._is_sliding[i]]
            x, kv_out = layer(x, cos, sin, kv_cache, i, layer_shared_kv, per_layer_input)
            if i in self._kv_provider_layers:
                shared_kv[self._is_sliding[i]] = kv_out
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
