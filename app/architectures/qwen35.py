import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.layers import RMSNorm
from app.architectures.qwen35_deltanet import Qwen35GatedDeltaNet
from app.architectures.qwen35_layers import (
    Qwen35Attention,
    Qwen35DecoderLayer,
    materialize_decoder_layer,
)
from app.architectures.rope import InterleavedMRopeEmbedding, RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.mamba_cache import NemotronHHybridCache

logger = logging.getLogger(__name__)


class Qwen35Architecture(ModelArchitecture):
    """Alibaba Qwen3.5 (dense) - a hybrid decoder: every `full_attention_interval`-th layer (3, 7,
    11, ... for interval 4) is gated full attention (`Qwen35Attention`), every other layer is a
    Gated DeltaNet linear-attention mixer (`Qwen35GatedDeltaNet`); all layers share a dense SwiGLU
    FFN. Hyperparameters and tensor names confirmed against the real header of
    `Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF` (range-fetched 2026-10-01): 32 layers, 16 heads /
    4 KV heads of dim 256, `rope.dimension_count=64`, 16 key / 32 value linear-attention heads of
    dim 128. `qwen35`'s own `ssm.*` keys are read here as: `group_count` = key heads,
    `time_step_rank` = value heads, `state_size` = per-head dim, `inner_size` = value dim.

    Cache: reuses `NemotronHHybridCache` (a KV cache for the attention layers, conv+recurrent state
    for the linear ones - linear layers are registered under its `"mamba"` kind). Recurrent state
    can't be rolled back, so `PromptCache` reuses a prefix across turns via snapshots captured
    during a (chunked) prefill - see its docstring.

    Vision: image embeddings (from `Qwen3VLVisionEncoder`) are spliced in at the image-pad
    positions, with 3-axis M-RoPE position ids supplied by the caller. Out of scope: the MTP
    head, and the MoE siblings (`qwen3next`/`qwen4exp`) - different `general.architecture` values.
    """

    NAME = "qwen35"
    SUPPORTS_BATCHED_DECODE = True  # see tests/unit/test_batched_decode.py
    SUPPORTS_VISION = True

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
        tied_embeddings: bool = False,
        tiled_heads: bool = True,
    ) -> None:
        super().__init__()
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        self._tied_embeddings = tied_embeddings
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_head_kv = metadata.get_u32(arch("attention.head_count_kv"), self.n_head)
        self.head_dim = metadata.get_u32(arch("attention.key_length"), self.n_embd // self.n_head)
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        self.rope_dim = metadata.get_u32(arch("rope.dimension_count"), self.head_dim)
        interval = metadata.get_u32(arch("full_attention_interval"), 4)
        self.layer_types = [
            "attention" if (i + 1) % interval == 0 else "mamba" for i in range(self.n_layer)
        ]
        self.conv_kernel = metadata.get_u32(arch("ssm.conv_kernel"))
        self.ssm_head_dim = metadata.get_u32(arch("ssm.state_size"))
        self.n_k_heads = metadata.get_u32(arch("ssm.group_count"))
        self.n_v_heads = metadata.get_u32(arch("ssm.time_step_rank"))
        self.conv_dim = (2 * self.n_k_heads + self.n_v_heads) * self.ssm_head_dim
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )

        self._rope_kwargs = dict(
            head_dim=self.rope_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self._rope_sections = metadata.get(arch("rope.dimension_sections"))
        self.rope = self._build_rope()
        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList([self._build_layer(t, tiled_heads) for t in self.layer_types])
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        if not tied_embeddings:
            self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False, dtype=dtype)

    def _build_layer(self, layer_type: str, tiled_heads: bool) -> Qwen35DecoderLayer:
        attention = linear = None
        if layer_type == "attention":
            attention = Qwen35Attention(
                self.n_embd, self.n_head, self.n_head_kv, self.head_dim, self.rms_eps, self._dtype
            )
        else:
            linear = Qwen35GatedDeltaNet(
                self.n_embd,
                self.n_k_heads,
                self.n_v_heads,
                self.ssm_head_dim,
                self.conv_kernel,
                self.rms_eps,
                tiled_heads,
                self._dtype,
            )
        return Qwen35DecoderLayer(
            self.n_embd, self.ffn_len, self.rms_eps, attention, linear, self._dtype
        )

    def _build_rope(self) -> RotaryEmbedding:
        """Interleaved M-RoPE when the GGUF has `rope.dimension_sections` (every real Qwen3.5
        does) - it matches plain RoPE for text; image tokens carry 3-axis positions."""
        if self._rope_sections:
            return InterleavedMRopeEmbedding(**self._rope_kwargs, sections=self._rope_sections)
        return RotaryEmbedding(**self._rope_kwargs)

    def _rebuild_derived_buffers(self) -> None:
        self.rope = self._build_rope()

    def build_cache(self, max_seq_len: int, dtype: torch.dtype) -> NemotronHHybridCache:
        cache = NemotronHHybridCache(
            layer_types=self.layer_types,
            attention_layer_shape=(self.n_head_kv, self.head_dim),
            mamba_conv_state_shape=(self.conv_kernel - 1, self.conv_dim),
            mamba_ssm_state_shape=(self.n_v_heads, self.ssm_head_dim, self.ssm_head_dim),
            max_seq_len=max_seq_len,
            dtype=dtype,
        )
        # The recurrent state accumulates over the whole context - keep it float32 even when
        # activations/KV are bf16.
        cache.ssm_state = [s.float() for s in cache.ssm_state]
        return cache

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "Qwen35Architecture":
        model = cls._construct_without_init(
            loader.metadata,
            dtype=dtype,
            enable_quantized_native=enable_quantized_native,
            tied_embeddings=not loader.has_tensor("output.weight"),
        )
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        enabled = self._enable_quantized_native
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        if not self._tied_embeddings:
            self.lm_head = self._load_projection(
                loader, "output.weight", self.lm_head, self._dtype, enabled
            )
        for i, layer in enumerate(self.layers):
            started = time.monotonic()
            materialize_decoder_layer(
                loader, f"blk.{i}.", layer, self._load_projection, self._dtype, enabled
            )
            logger.debug(
                "loading weights: layer %d/%d (%s) in %.1fs",
                i + 1,
                self.n_layer,
                self.layer_types[i],
                time.monotonic() - started,
            )
            if stop_check is not None and stop_check():
                raise GenerationCancelledError(f"stopped loading weights at layer {i + 1}")

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: NemotronHHybridCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        _, seq_len = input_ids.shape
        if kv_cache is None:  # plain full-sequence pass (tests/oracle): a throwaway cache
            kv_cache = self.build_cache(seq_len, self._dtype)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        if getattr(kv_cache, "batched", False):  # batched decode: one text position per sequence
            cos, sin = RotaryEmbedding.forward(self.rope, position_ids)
        else:
            cos, sin = self.rope(position_ids)  # (T,) text, or (3, T) with images

        x = self.token_embd(input_ids)
        for start, embeds in image_embeddings or []:
            x[:, start : start + embeds.shape[0], :] = embeds.to(x.dtype)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_cache, i)
            if stop_check is not None and stop_check():
                logger.info("generation stop requested - cancelling after layer %d", i + 1)
                raise GenerationCancelledError(f"stopped after layer {i + 1}/{self.n_layer}")

        if self._last_logits_only:
            # Only the last position's logits are read; a full-sequence LM head would dequantize
            # the whole vocab matrix and build a (T, vocab) row block per prefill piece.
            x = x[:, -1:, :]
        x = self.output_norm(x)
        if self._tied_embeddings:
            return F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        return self.lm_head(x)
