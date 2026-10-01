import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.falcon_layers import FalconDecoderLayer
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class FalconArchitecture(ModelArchitecture):
    """TII Falcon (real `tiiuae/falcon-7b` shape, confirmed 2026-09-30 against a real downloaded

    GGUF header) - a parallel-residual GQA/MQA decoder (see `FalconDecoderLayer`'s own docstring
    for the exact real formula/verification). Real, confirmed deltas from every other
    architecture here:
    - Real fused `attn_qkv` tensor (no separate `attn_q`/`attn_k`/`attn_v`) - real MQA layout
      (`head_count_kv=1`), split at materialize time into contiguous
      `[q rows | k rows | v rows]` blocks (llama.cpp's own real converter rearranges HF's native
      per-head-interleaved layout into exactly this - confirmed via the real file's own
      `falcon.tensor_data_layout = "jploski"` metadata key, which names that rearrangement). No
      bias on it (`config.bias = False`) - see `FalconMLP`'s own docstring for the same real
      finding on the MLP side.
    - Real biased `attn_norm` (the ONE shared parallel-residual norm) despite `config.bias`
      being unrelated to it - `nn.LayerNorm`'s own bias is gated by nothing but its own default,
      confirmed via a real `attn_norm.bias` tensor.
    - No `rope.freq_base`/`vocab_size` metadata key on this real file - falls back to HF's own
      real RoPE default (10000.0) and to `len(tokenizer.ggml.tokens)` respectively, the same
      fallback pattern `Phi2Architecture`/`LlamaArchitecture` already established.

    Deliberately out of scope for this pass, not silently assumed to be the same shape: the
    newer `new_decoder_architecture` variant (a real second `attn_norm_2` tensor, independent
    norms per branch instead of one shared one) and the older `tiiuae/falcon-rw-*` variant
    (`parallel_attn=False` - sequential, not parallel - and real ALiBi instead of RoPE, a
    genuinely different positional-encoding mechanism not implemented anywhere in this repo) -
    both real, confirmed-via-source variants, tracked as separate future work.
    """

    NAME = "falcon"
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
        self.n_head_kv = metadata.get_u32(arch("attention.head_count_kv"), self.n_head)
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )

        self._rope_kwargs = dict(
            head_dim=self.head_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"), 10000.0)
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                FalconDecoderLayer(
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
    ) -> "FalconArchitecture":
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

        q_rows = self.n_head * self.head_dim
        kv_rows = self.n_head_kv * self.head_dim
        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            layer.input_layernorm.bias.copy_(loader.load_tensor(prefix + "attn_norm.bias"))

            # attn_qkv is one real fused, bias-free (out_rows, n_embd) tensor - llama.cpp's own
            # converter already rearranges it into contiguous [q rows | k rows | v rows] blocks
            # (real `falcon.tensor_data_layout = "jploski"` metadata names this) rather than HF's
            # native per-head-interleaved layout - a plain row-range split, no per-head reshuffle
            # needed. Always today's exact path, same reasoning Phi2Architecture's own comment
            # gives for not attempting to split a packed/quantized tensor's raw bytes this round.
            qkv_weight = loader.load_tensor(prefix + "attn_qkv.weight")
            layer.self_attn.q_proj.weight.copy_(qkv_weight[:q_rows])
            layer.self_attn.k_proj.weight.copy_(qkv_weight[q_rows : q_rows + kv_rows])
            layer.self_attn.v_proj.weight.copy_(qkv_weight[q_rows + kv_rows : q_rows + 2 * kv_rows])

            enabled = self._enable_quantized_native
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, self._dtype, enabled
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
        del image_embeddings  # no real vision-language Falcon checkpoint exists to fuse
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
