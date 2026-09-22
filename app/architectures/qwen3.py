import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.layers import RMSNorm
from app.architectures.qwen_layers import QwenDecoderLayer
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Qwen3Architecture(ModelArchitecture):
    """Alibaba Qwen3 - `qwen2`'s shape (plain GQA/RoPE/SwiGLU) with no bias anywhere (confirmed:
    real Qwen3 GGUFs have no `attn_*.bias` tensors at all, not just zeroed - `config.attention_bias
    = False` on every real small checkpoint) plus one real structural addition confirmed against
    HF `transformers`' own `modeling_qwen3.py` (2026-09-22): **QK-norm** - a per-head-dim `RMSNorm`
    applied to both q and k right after the reshape-to-heads split, before RoPE (see
    `QwenAttention`'s own docstring for the exact placement).

    `head_dim` **must** be read from real GGUF metadata (`qwen3.attention.key_length`) rather than
    derived from `n_embd // n_head` - real Qwen3-0.6B has `hidden_size=1024, num_attention_heads
    =16` (naive division gives 64) but a real `head_dim=128`, confirmed via both HF's own
    config.json and live GGUF metadata bytes range-read from a real downloaded file. This
    architecture's own version of Granite's HF-vs-GGUF naming trap - the same defensive
    `get_u32(..., default=...)` read this project already uses handles both this case (falls
    through to the real value, present) and `qwen2`'s (falls back to the derived value, since the
    key is typically absent there) with the identical code.

    **None of `attn_q.weight`/`attn_k.weight`/`attn_q_norm.weight`/`attn_k_norm.weight` need
    `unpermute_rope_rows`** - the reasoning that they would (llama.cpp's own C++ pipeline applies
    `q_norm` and RoPE directly to the still-GGUF-native-ordered tensor, so matricxon undoing that
    ordering would need `q_norm`'s weight undone identically) turned out to rest on a false
    premise: real-weight oracle validation on `qwen2` (this architecture's sibling, same shared
    `unpermute_rope_rows`-listing docstring claim) proved empirically that real Qwen weights don't
    need this reordering at all - applying it silently turned a working model into one that only
    ever generated garbage (confirmed live: never predicted "Paris" after "The capital of France
    is", despite every other check - tokenizer round-trip, KV-cache self-consistency, rope_theta -
    passing cleanly). `unpermute_rope_rows`'s own docstring listing Qwen as a model that needs it
    was simply wrong for this architecture family. Fixed the same way here, verified against
    `scripts/oracle/validate_qwen3.py`.
    """

    NAME = "qwen3"

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
        tied_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        self._tied_embeddings = tied_embeddings
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_head_kv = metadata.get_u32(arch("attention.head_count_kv"))
        self.head_dim = metadata.get_u32(arch("attention.key_length"), self.n_embd // self.n_head)
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )

        self._rope_kwargs = dict(
            head_dim=self.head_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                QwenDecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.ffn_len,
                    self.rms_eps,
                    qkv_bias=False,
                    qk_norm_eps=self.rms_eps,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        if not tied_embeddings:
            self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False, dtype=dtype)

    def _rebuild_derived_buffers(self) -> None:
        self.rope = RotaryEmbedding(**self._rope_kwargs)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        return [(self.n_head_kv, self.head_dim)] * self.n_layer

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "Qwen3Architecture":
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
        stage_started = time.monotonic()
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        if not self._tied_embeddings:
            self.lm_head = self._load_projection(
                loader, "output.weight", self.lm_head, self._dtype, self._enable_quantized_native
            )
        logger.debug(
            "materializing: token_embd + output_norm + lm_head in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            layer.post_attention_layernorm.weight.copy_(
                loader.load_tensor(prefix + "ffn_norm.weight")
            )
            layer.self_attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
            layer.self_attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
            layer.self_attn.q_norm.weight.copy_(loader.load_tensor(prefix + "attn_q_norm.weight"))
            layer.self_attn.k_norm.weight.copy_(loader.load_tensor(prefix + "attn_k_norm.weight"))
            enabled = self._enable_quantized_native
            layer.self_attn.v_proj = self._load_projection(
                loader, prefix + "attn_v.weight", layer.self_attn.v_proj, self._dtype, enabled
            )
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, self._dtype, enabled
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
        del image_embeddings  # no real vision-language Qwen3 checkpoint exists yet to fuse
        _, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        cos, sin = self.rope(position_ids)

        x = self.token_embd(input_ids)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_cache, i)
            if stop_check is not None and stop_check():
                logger.info(
                    "generation stop requested - cancelling after layer %d/%d", i + 1, self.n_layer
                )
                raise GenerationCancelledError(f"stopped after layer {i + 1}/{self.n_layer}")

        x = self.output_norm(x)
        if self._tied_embeddings:
            return F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        return self.lm_head(x)
