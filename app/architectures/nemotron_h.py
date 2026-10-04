import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.layers import RMSNorm
from app.architectures.nemotron_h_layers import (
    build_nemotron_h_layer,
    materialize_attention_layer,
    materialize_mamba_layer,
    materialize_mlp_layer,
)
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.mamba_cache import NemotronHHybridCache

logger = logging.getLogger(__name__)


class NemotronHArchitecture(ModelArchitecture):
    """NVIDIA Nemotron-H - a real hybrid decoder with THREE interleaved per-layer types (not a
    uniform stack, unlike every other architecture in this repo): Mamba-2 (SSM), plain GQA
    attention (no RoPE at all), and plain non-gated squared-ReLU MLP. Confirmed against HF
    `transformers`' real `modeling_nemotron_h.py`, llama.cpp's real shipped `nemotron_h` GGUF
    support, and a real downloaded `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` file's own header
    (range-fetched directly, 2026-09-22 - no full multi-GB download needed).

    Layer type is derived per real layer index from two existing GGUF **per-layer arrays**
    (`attention.head_count_kv`/`feed_forward_length` - confirmed real on the target checkpoint,
    42 entries each), not a dedicated "layer type" key - `mamba` iff both are 0, `attention` iff
    `head_count_kv != 0`, `mlp` otherwise (llama.cpp's own real C++ loader logic, replicated here
    in Python). Real target checkpoint: 21 Mamba-2 / 4 attention / 17 MLP layers. Per-layer
    construction/materialization is dispatched to `build_nemotron_h_layer`/`materialize_*_layer`
    (`app/architectures/nemotron_h_layers.py`), kept next to the layer classes themselves so this
    file stays focused on metadata parsing and orchestration.

    Uses its own hybrid cache (`NemotronHHybridCache`, `app/runtime/mamba_cache.py`) rather than
    the plain per-attention-layer `KVCache` every other architecture uses - see
    `ModelArchitecture.build_cache`'s own docstring for why this is a clean override rather than a
    special case, and why `kv_cache_layer_shapes` is simply never defined here (it was never
    ABC-enforced to begin with).

    Quantized-native compute (`app/models/load_dtype.py`'s `QUANTIZED_NATIVE_WIRED_ARCHITECTURES`/
    `_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH`) covers `attn_v`/`attn_output`/`ffn_up`/`ffn_down`/
    `output.weight`, plus the Mamba mixer's own `ssm_in`/`ssm_out` (2026-09-29) - real, non-marginal
    savings confirmed on an installed 12B checkpoint (MLP tensors are 47.7% of real total elements
    there, and `ssm_in`/`ssm_out` alone are 99.9% of the SSM bucket's own elements - together this
    now covers nearly the whole model; estimated RAM for that file dropped ~29.5GB -> ~18.6GB).
    Every other SSM tensor genuinely isn't 2D `nn.Linear`-shaped (`ssm_a`/`ssm_d`/`ssm_dt.bias` are
    1D, `ssm_conv1d` is a depthwise conv kernel, `ssm_norm` a grouped-norm weight) and stays a
    plain `.copy_()` into its raw `nn.Parameter`, same as `token_embd` and `attn_q`/`attn_k` (this
    architecture has no RoPE, so there's no packed-row permutation reason to route those two
    through `_load_projection` the way llama/gemma4 do) - a real, separate follow-up, not
    attempted here.
    """

    NAME = "nemotron_h"
    SUPPORTS_GPU = True

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
        assert isinstance(self.n_head, int), (
            "nemotron_h.attention.head_count must be a scalar, not a per-layer array - "
            "confirmed scalar on the real target checkpoint"
        )
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )

        # The one genuinely new kind of metadata read this architecture needs: real per-layer
        # arrays (not scalars) that llama.cpp's own converter uses to encode which of the three
        # real layer types each real layer index actually is.
        head_count_kv_arr = metadata.get_array(arch("attention.head_count_kv"))
        ffn_len_arr = metadata.get_array(arch("feed_forward_length"))
        self.layer_types: list[str] = []
        self.n_head_kv_per_layer: list[int] = []
        self.ffn_len_per_layer: list[int] = []
        for i in range(self.n_layer):
            hkv, ffn = head_count_kv_arr[i], ffn_len_arr[i]
            if hkv == 0 and ffn == 0:
                self.layer_types.append("mamba")
            elif hkv != 0:
                self.layer_types.append("attention")
            else:
                self.layer_types.append("mlp")
            self.n_head_kv_per_layer.append(hkv)
            self.ffn_len_per_layer.append(ffn)
        # Real attention layers in this checkpoint share one n_head_kv value - kept for
        # NemotronHHybridCache's uniform attention_layer_shape.
        self.n_head_kv = next(h for h in self.n_head_kv_per_layer if h != 0)

        # ssm.* namespace - generic, shared with Mamba/Mamba2/Jamba, not nemotron_h-specific.
        self.conv_kernel = metadata.get_u32(arch("ssm.conv_kernel"))
        self.mamba_d_inner = metadata.get_u32(arch("ssm.inner_size"))
        self.d_state = metadata.get_u32(arch("ssm.state_size"))
        # Repurposed key: holds the real Mamba-2 head COUNT, not a "rank" (llama.cpp naming
        # quirk carried over from Mamba-1) - confirmed against the real target checkpoint.
        self.mamba_num_heads = metadata.get_u32(arch("ssm.time_step_rank"))
        self.n_group = metadata.get_u32(arch("ssm.group_count"))
        self.mamba_head_dim = self.mamba_d_inner // self.mamba_num_heads

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                build_nemotron_h_layer(
                    self.layer_types[i],
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv_per_layer[i],
                    self.head_dim,
                    self.ffn_len_per_layer[i],
                    self.rms_eps,
                    self.mamba_d_inner,
                    self.mamba_num_heads,
                    self.mamba_head_dim,
                    self.d_state,
                    self.n_group,
                    self.conv_kernel,
                    dtype,
                )
                for i in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        if not tied_embeddings:
            self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False, dtype=dtype)

    def build_cache(self, max_seq_len: int, dtype: torch.dtype) -> NemotronHHybridCache:
        return NemotronHHybridCache(
            layer_types=self.layer_types,
            attention_layer_shape=(self.n_head_kv, self.head_dim),
            mamba_conv_state_shape=(
                self.conv_kernel - 1,
                self.mamba_d_inner + 2 * self.n_group * self.d_state,
            ),
            mamba_ssm_state_shape=(self.mamba_num_heads, self.mamba_head_dim, self.d_state),
            max_seq_len=max_seq_len,
            dtype=dtype,
            device=self.device,
        )

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "NemotronHArchitecture":
        model = cls._construct_without_init(
            loader.metadata,
            dtype=dtype,
            enable_quantized_native=enable_quantized_native,
            tied_embeddings=not loader.has_tensor("output.weight"),
        )
        # No _rebuild_derived_buffers override needed - no RoPE, no derived buffer to recreate.
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        if not self._tied_embeddings:
            self.lm_head = self._load_projection(
                loader, "output.weight", self.lm_head, self._dtype, self._enable_quantized_native
            )

        enabled = self._enable_quantized_native
        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))

            layer_type = self.layer_types[i]
            if layer_type == "mamba":
                materialize_mamba_layer(
                    loader, prefix, layer.mixer, self._load_projection, self._dtype, enabled
                )
            elif layer_type == "attention":
                materialize_attention_layer(
                    loader, prefix, layer.self_attn, self._load_projection, self._dtype, enabled
                )
            else:
                materialize_mlp_layer(
                    loader, prefix, layer.mlp, self._load_projection, self._dtype, enabled
                )

            logger.debug(
                "loading weights: layer %d/%d (%s, not computing yet) in %.1fs",
                i + 1,
                self.n_layer,
                layer_type,
                time.monotonic() - layer_started,
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
        del image_embeddings  # no real vision-language Nemotron-H checkpoint exists yet to fuse
        del position_ids  # no RoPE anywhere in this architecture - nothing needs a position id

        _, seq_len = input_ids.shape
        x = self.token_embd(input_ids)
        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x, kv_cache, i)
            # Real per-layer-type timing (mamba/attention/mlp) - a real gap this closes
            # (2026-09-29): _forward_impl logged nothing here before, unlike gemma4's own
            # "attention"/"ffn" lines. Each real layer here is only ever one stage, so one line
            # with its type carries what gemma4 needs two nested lines for.
            logger.debug(
                "layer %d/%d (%s): %.1fms (seq_len=%d)",
                i + 1,
                self.n_layer,
                self.layer_types[i],
                (time.monotonic() - layer_started) * 1000,
                seq_len,
            )
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
