import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.command_r_layers import CommandRDecoderLayer
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class CommandRArchitecture(ModelArchitecture):
    """Cohere Command-R - a plain GQA/RoPE decoder with a real parallel attention+FFN block
    (confirmed against HF `transformers`' own `modeling_cohere.py`, 2026-09-22 - see
    `CommandRDecoderLayer`'s own docstring for the exact formula and reuse story).

    Real, confirmed deltas from every other architecture in this repo:
    - `attn_norm`/`output_norm` are real, bias-free `nn.LayerNorm` (mean-centered), not `RMSNorm` -
      the real GGUF metadata key itself signals this (`attention.layer_norm_epsilon`, not the
      `...layer_norm_rms_epsilon` every RMSNorm-based architecture here uses).
    - No bias anywhere, no QK-norm (QK-norm only exists on the 104B Command-R+, out of scope - see
      `command_r_layers.py`'s own docstring).
    - **No `unpermute_rope_rows` needed** - confirmed via real source evidence, not assumed:
      llama.cpp's real `conversion/command_r.py` has zero permute calls, and HF's own
      `modeling_cohere.py` explicitly comments its `rotate_half` as *"different from e.g. Llama"* -
      Cohere's rotate-half is already consecutive-pair/interleaved, matching ggml's native layout
      directly. Same real finding as `qwen2`/`qwen3` (also confirmed to need no permutation) - a
      second, independent case proving RoPE-family resemblance alone is never a valid signal for
      whether this reordering applies (`qwen2`'s own docstring has the full story of what
      happened when that assumption was trusted without verification).
    - **Logit scaling multiplies, not divides**: `logits = lm_head(x) * logit_scale` (unlike
      Granite's `logits_scaling`, which divides) - real GGUF key `logit_scale`, real default
      `0.0625`.
    - **Every real file ships tied embeddings** - llama.cpp's real GGUF writer never emits a
      separate `output.weight` tensor for this architecture at all. The existing dynamic
      `loader.has_tensor("output.weight")` detection (same as every other architecture here)
      needs no special-casing to get this right - it naturally resolves tied for any real file.

    Real `rope.freq_base` for a real deployed checkpoint is `10000.0`, not HF's config-class
    default of `500000.0` - as always, read from real GGUF metadata, never hardcode a class
    default.

    `cohere2` (Command-R7B - a real, distinct GGUF architecture string with a genuinely different
    sliding-window/full-attention hybrid structure) and Command-R+ (104B, needs QK-norm, no real
    checkpoint small enough for this project's target hardware) are both deliberately out of
    scope for this pass - tracked as separate future work, not folded in here.
    """

    NAME = "command-r"

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
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
        self.logit_scale = metadata.get_f32(arch("logit_scale"), 1.0)
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
                CommandRDecoderLayer(
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
        self.output_norm = nn.LayerNorm(
            self.n_embd, eps=self.layer_norm_eps, bias=False, dtype=dtype
        )
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
    ) -> "CommandRArchitecture":
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
            # No unpermute_rope_rows anywhere - confirmed not needed for this architecture, see
            # this class's own docstring.
            layer.self_attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
            layer.self_attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
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
        del image_embeddings  # no real vision-language Command-R checkpoint exists yet to fuse
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
            logits = F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        else:
            logits = self.lm_head(x)
        return logits * self.logit_scale
