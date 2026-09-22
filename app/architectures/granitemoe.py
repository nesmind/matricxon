import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.granite_layers import GraniteDecoderLayer
from app.architectures.granitemoe_layers import GraniteMoeFFN
from app.architectures.layers import RMSNorm, unpermute_rope_rows
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class GraniteMoeArchitecture(ModelArchitecture):
    """GraniteMoE - `GraniteArchitecture`'s attention block and four "Power"-scaling multipliers,
    unchanged, with a sparse Mixture-of-Experts FFN (`GraniteMoeFFN`) in place of dense Granite's
    `SwiGLUMLP` (confirmed against HF `transformers`' `modeling_granitemoe.py`, 2026-09-22: only
    the FFN/MLP block differs from dense Granite - attention stays plain GQA).

    Two extra real GGUF metadata keys beyond dense Granite's: `granitemoe.expert_count`
    (`num_local_experts`) and `granitemoe.expert_used_count` (`num_experts_per_tok`) - llama.cpp's
    standard generic MoE key pair, the same one Mixtral/DeepSeek-MoE/Qwen-MoE etc. all use.

    Deliberately **not** wired into quantized-native compute (`_load_projection`) for its expert
    tensors: `QuantizedLinear`/the fused GEMV kernels are hard-assumed 2D `(out_features,
    in_features)` throughout (confirmed by reading `app/architectures/base.py`'s
    `_load_projection` - a real 3D `(num_experts, ...)` tensor crashes its `out_features,
    in_features = shape` unpack), so every MoE tensor here (router included, for one consistent
    code path rather than mixed logic) goes through a plain `loader.load_tensor()` `.copy_()`,
    same as it would with quantized-native compute off anyway. `"granitemoe"` is deliberately not
    added to `QUANTIZED_NATIVE_WIRED_ARCHITECTURES`/`_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH`
    (`app/models/load_dtype.py`) - a real, permanent scope boundary until a batched/grouped
    quantized MoE kernel exists, not an oversight.
    """

    NAME = "granitemoe"

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
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )
        self.num_experts = metadata.get_u32(arch("expert_count"))
        self.num_experts_per_tok = metadata.get_u32(arch("expert_used_count"))

        self.attention_scale = metadata.get_f32(arch("attention.scale"))
        self.embedding_multiplier = metadata.get_f32(arch("embedding_scale"), 1.0)
        self.residual_multiplier = metadata.get_f32(arch("residual_scale"), 1.0)
        self.logits_scaling = metadata.get_f32(arch("logit_scale"), 1.0)

        self._rope_kwargs = dict(
            head_dim=self.head_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                GraniteDecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.rms_eps,
                    self.residual_multiplier,
                    self.attention_scale,
                    mlp=GraniteMoeFFN(
                        self.n_embd,
                        self.ffn_len,
                        self.num_experts,
                        self.num_experts_per_tok,
                        dtype=dtype,
                    ),
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
    ) -> "GraniteMoeArchitecture":
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
            layer.self_attn.q_proj.weight.copy_(
                unpermute_rope_rows(loader.load_tensor(prefix + "attn_q.weight"), self.n_head)
            )
            layer.self_attn.k_proj.weight.copy_(
                unpermute_rope_rows(loader.load_tensor(prefix + "attn_k.weight"), self.n_head_kv)
            )
            enabled = self._enable_quantized_native
            layer.self_attn.v_proj = self._load_projection(
                loader, prefix + "attn_v.weight", layer.self_attn.v_proj, self._dtype, enabled
            )
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, self._dtype, enabled
            )
            # MoE tensors: always a plain .copy_() - see this class's own docstring for why
            # _load_projection (the quantized-native-eligible path) never applies here.
            layer.mlp.router.weight.copy_(loader.load_tensor(prefix + "ffn_gate_inp.weight"))
            layer.mlp.gate_exps.copy_(loader.load_tensor(prefix + "ffn_gate_exps.weight"))
            layer.mlp.up_exps.copy_(loader.load_tensor(prefix + "ffn_up_exps.weight"))
            layer.mlp.down_exps.copy_(loader.load_tensor(prefix + "ffn_down_exps.weight"))
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
        del image_embeddings  # no real vision-language GraniteMoE checkpoint exists yet to fuse
        _, seq_len = input_ids.shape
        logger.debug("forward start: seq_len=%d", seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        cos, sin = self.rope(position_ids)

        x = self.token_embd(input_ids) * self.embedding_multiplier

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x, cos, sin, kv_cache, i)
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

        x = self.output_norm(x)
        if self._tied_embeddings:
            logits = F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        else:
            logits = self.lm_head(x)
        return logits / self.logits_scaling
