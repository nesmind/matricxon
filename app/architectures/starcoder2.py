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
    """BigCode StarCoder2: a plain GQA/RoPE decoder (HF `modeling_starcoder2.py`). Deltas from the
    other GQA/RoPE decoders here:
    - A bias on every projection (`q_proj`/`k_proj`/`v_proj`/`o_proj` and the MLP's
      `up_proj`/`down_proj`): `config.use_bias` gates all (default `True` on every released
      checkpoint; no GGUF metadata key, so bias presence follows tensor existence, hardcoded `True`
      like `qwen2`'s `qkv_bias`).
    - A plain bias-affine `nn.LayerNorm`, not `RMSNorm` (key `attention.layer_norm_epsilon`, as
      `command_r.py`, whose LayerNorm is bias-free).
    - A plain non-gated MLP (`ffn_up`/`ffn_down`, no `ffn_gate`) with tanh-approximated GELU
      (`Starcoder2MLP`).
    - Sequential (not parallel) residual: the two-block pre-norm shape of `QwenDecoderLayer`.

    No `unpermute_rope_rows`, from source: llama.cpp's `conversion/starcoder.py` registers
    `StarCoder2Model` with no overrides (no `modify_tensors` or permute call, unlike
    `LlamaModel.permute`). A third case (after `qwen2`/`qwen3` and `command-r`) showing RoPE-family
    resemblance is never sufficient evidence (see `unpermute_rope_rows`).

    `bigcode/starcoder2-3b/-7b/-15b` tie embeddings, detected via
    `loader.has_tensor("output.weight")`, not `Starcoder2Config`'s `tie_word_embeddings=True`
    default.

    Validated on `second-state/StarCoder2-3B-GGUF` (Q4_K_M, bf16,
    `tools/manual_generate_check_starcoder2.py`): `"def fibonacci(n):"` completed into valid Python,
    confirming no `unpermute_rope_rows` (a wrong choice gives garbage, as with `qwen2`).
    `tokenizer.ggml.pre` is unset on this file; `GGUFTokenizer`'s default GPT-2 pre-tokenizer regex
    suffices.
    """

    NAME = "starcoder2"
    SUPPORTS_GPU = True
    SUPPORTS_BATCHED_DECODE = True

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
        # Falls back to n_head when absent (no GQA); see LlamaArchitecture.
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
            # No unpermute_rope_rows: not needed here (class docstring). q/k stay on the plain float
            # path, as every other architecture (only v/o/mlp can be quantized-native).
            layer.self_attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
            layer.self_attn.q_proj.bias.copy_(loader.load_tensor(prefix + "attn_q.bias"))
            layer.self_attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
            layer.self_attn.k_proj.bias.copy_(loader.load_tensor(prefix + "attn_k.bias"))
            enabled = self._enable_quantized_native
            # A bias on each of these (class docstring): the first reuse of `_load_projection`'s
            # `bias_tensor_name` outside `phi2.py`.
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
        del image_embeddings
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
