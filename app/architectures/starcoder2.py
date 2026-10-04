import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.rope import RotaryEmbedding
from app.architectures.starcoder2_layers import Starcoder2DecoderLayer
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Starcoder2Architecture(ModelArchitecture):
    """BigCode StarCoder2 - a plain GQA/RoPE decoder (confirmed against HF `transformers`' own

    `modeling_starcoder2.py`, 2026-09-29), real, confirmed deltas from every GQA/RoPE decoder
    already in this repo:
    - Real bias on **every** projection (`q_proj`/`k_proj`/`v_proj`/`o_proj` and the MLP's
      `up_proj`/`down_proj`) - `config.use_bias` gates all of them identically (real default
      `True` on every released checkpoint; not exposed as its own GGUF metadata key - bias
      presence is entirely tensor-existence-driven, same as every other architecture here,
      hardcoded `True` since no real released checkpoint sets it any other way, the same
      precedent `qwen2`'s own docstring already established for its own hardcoded `qkv_bias`).
    - Real, plain (bias-affine) `nn.LayerNorm`, not `RMSNorm` - the real GGUF metadata key itself
      signals this (`attention.layer_norm_epsilon`, the same key `command_r.py` uses for its own
      `LayerNorm`), unlike `command_r`'s bias-free variant: StarCoder2's real norm keeps its bias.
    - A plain, non-gated MLP (`ffn_up`/`ffn_down`, no `ffn_gate`) using the **tanh approximation**
      of GELU - see `Starcoder2MLP`'s own docstring.
    - Sequential (not parallel) residual - same two-block pre-norm shape as `QwenDecoderLayer`.

    **No `unpermute_rope_rows` needed** - confirmed via real source evidence, not assumed:
    llama.cpp's real `conversion/starcoder.py` registers `StarCoder2Model` with zero overrides
    beyond `model_arch` (no `modify_tensors`, no permute call, unlike `LlamaModel`'s own
    `permute` method) - a third real, independently-confirmed case (after `qwen2`/`qwen3` and
    `command-r`) that RoPE-family resemblance alone is never sufficient evidence either way
    (`unpermute_rope_rows`'s own docstring has the full history of what happened when that
    assumption was trusted without verification).

    Real checkpoints (`bigcode/starcoder2-3b/-7b/-15b`) tie embeddings - detected dynamically
    (`loader.has_tensor("output.weight")`), same as every other architecture here, not assumed
    from `Starcoder2Config`'s own `tie_word_embeddings=True` class default.

    **Real-weight validated (2026-09-29)** against a real downloaded
    `second-state/StarCoder2-3B-GGUF` (Q4_K_M, bf16 forward, via
    `scripts/manual_generate_check_starcoder2.py`): `"def fibonacci(n):"` greedily completed into
    coherent, syntactically valid Python (`"\n    if n == 0:\n        return 1\n    elif n == 1
    or n <"`) - real, working confirmation of the "no `unpermute_rope_rows`" finding above (wrong
    either way here would have produced garbage, the same signal that caught the real `qwen2`
    bug). `tokenizer.ggml.pre` is unset on this real file - the plain GPT-2 pre-tokenizer regex
    `GGUFTokenizer` already defaults to needs no per-architecture override here, unlike Llama 3's.
    """

    NAME = "starcoder2"
    SUPPORTS_GPU = True
    SUPPORTS_BATCHED_DECODE = True  # see tests/unit/test_batched_decode.py

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
        # Falls back to n_head when absent (no GQA) - see LlamaArchitecture's own docstring for
        # the real file that confirmed this gap.
        self.n_head_kv = metadata.get_u32(arch("attention.head_count_kv"), self.n_head)
        self.head_dim = metadata.get_u32(arch("attention.key_length"), self.n_embd // self.n_head)
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
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
                Starcoder2DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.ffn_len,
                    self.layer_norm_eps,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.output_norm = nn.LayerNorm(self.n_embd, eps=self.layer_norm_eps, dtype=dtype)
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
    ) -> "Starcoder2Architecture":
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
        self.output_norm.bias.copy_(loader.load_tensor("output_norm.bias"))
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
            layer.input_layernorm.bias.copy_(loader.load_tensor(prefix + "attn_norm.bias"))
            layer.post_attention_layernorm.weight.copy_(
                loader.load_tensor(prefix + "ffn_norm.weight")
            )
            layer.post_attention_layernorm.bias.copy_(loader.load_tensor(prefix + "ffn_norm.bias"))
            # No unpermute_rope_rows anywhere - confirmed not needed for this architecture, see
            # this class's own docstring. q/k stay on the plain float path always, same as every
            # other architecture here (only v/o/mlp are ever quantized-native eligible).
            layer.self_attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
            layer.self_attn.q_proj.bias.copy_(loader.load_tensor(prefix + "attn_q.bias"))
            layer.self_attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
            layer.self_attn.k_proj.bias.copy_(loader.load_tensor(prefix + "attn_k.bias"))
            enabled = self._enable_quantized_native
            # Real bias on every one of these (see this class's own docstring) - the first real
            # reuse of `_load_projection`'s `bias_tensor_name` param outside `phi2.py`.
            layer.self_attn.v_proj = self._load_projection(
                loader,
                prefix + "attn_v.weight",
                layer.self_attn.v_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "attn_v.bias",
            )
            layer.self_attn.o_proj = self._load_projection(
                loader,
                prefix + "attn_output.weight",
                layer.self_attn.o_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "attn_output.bias",
            )
            layer.mlp.up_proj = self._load_projection(
                loader,
                prefix + "ffn_up.weight",
                layer.mlp.up_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "ffn_up.bias",
            )
            layer.mlp.down_proj = self._load_projection(
                loader,
                prefix + "ffn_down.weight",
                layer.mlp.down_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "ffn_down.bias",
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
        del image_embeddings  # no real vision-language StarCoder2 checkpoint exists to fuse
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

        if self._last_logits_only:
            x = x[:, -1:, :]
        x = self.output_norm(x)
        if self._tied_embeddings:
            return F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        return self.lm_head(x)
