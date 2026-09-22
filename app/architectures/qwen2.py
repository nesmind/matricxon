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


class Qwen2Architecture(ModelArchitecture):
    """Alibaba Qwen2 - a plain GQA/RoPE/SwiGLU decoder, the same shape `llama`/`granite` already
    are, plus one real delta confirmed against HF `transformers`' own `modeling_qwen2.py`
    (2026-09-22): `q_proj`/`k_proj`/`v_proj` carry a real bias (hardcoded `bias=True` in HF's own
    source, not config-gated) while `o_proj` never does - confirmed against a real downloaded
    `Qwen2.5-0.5B-Instruct-GGUF` file's own tensor names (`attn_q.bias`/`attn_k.bias`/
    `attn_v.bias` present, no `attn_output.bias`).

    **Neither the real q/k weight NOR their bias need `unpermute_rope_rows` at all** - confirmed
    empirically (2026-09-22) against a real downloaded `Qwen2.5-0.5B-Instruct-GGUF` file (both
    Q4_K_M and F16) and its real HF `Qwen2.5-0.5B-Instruct` safetensors counterpart. This
    contradicts `unpermute_rope_rows`'s own docstring, which lists Qwen among the models needing
    it - that claim was never true for real Qwen2 GGUFs (llama.cpp's real converter evidently
    does not apply the rotate-half row-interleave to this architecture's q/k projections, the same
    way `gemma4`'s own converter doesn't - see that architecture's own comment). Applying it
    anyway silently produced a real, severe bug: real generation was pure garbage (repetitive,
    ungrammatical tokens, e.g. never predicting "Paris" after "The capital of France is"), even
    though every other check (tokenizer round-trip, KV-cache-vs-no-cache self-consistency, `rope
    .freq_base`, tied-embedding detection) passed cleanly - a textbook "runs without error but
    produces wrong output" bug, exactly the class of mistake real-weight oracle validation exists
    to catch, caught here by comparing the raw (unmodified) GGUF weight against HF's real
    `q_proj.weight` directly: cosine ~0.90 raw vs. `-0.04` (anti-correlated) after applying
    `unpermute_rope_rows`. Both `attn_q.weight`/`attn_k.weight` and their bias tensors are loaded
    with a plain `.copy_()`, no reordering.

    Real `head_dim` is usually absent from a real Qwen2 GGUF's metadata (falls back to
    `n_embd // n_head`, HF's own default) - read defensively via `get_u32` with that fallback
    rather than assumed, the same defensive pattern `llama`'s vocab_size fallback established.
    Real Qwen2.5 checkpoints tie embeddings up to 3B and stop tying at 7B - detected dynamically
    (`loader.has_tensor("output.weight")`) rather than assumed by size, same as `llama`/`granite`.
    """

    NAME = "qwen2"

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
                    qkv_bias=True,
                    qk_norm_eps=None,
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
    ) -> "Qwen2Architecture":
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
            layer.self_attn.q_proj.bias.copy_(loader.load_tensor(prefix + "attn_q.bias"))
            layer.self_attn.k_proj.bias.copy_(loader.load_tensor(prefix + "attn_k.bias"))
            layer.self_attn.v_proj.bias.copy_(loader.load_tensor(prefix + "attn_v.bias"))
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
        del image_embeddings  # no real vision-language Qwen2 checkpoint exists yet to fuse
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
