"""Loading weights that stay packed (quantized) for the model's whole lifetime; split from base.py,
from which every
`ModelArchitecture` inherits this.

`_load_projection` covers per-layer projections, including attn_q/attn_k with their
`unpermute_rope_rows` row permutation
(applied to the packed rows, `PackedRows`), and `_load_token_embedding` the token-embedding table
(`QuantizedEmbedding`).
"""

import torch
from torch import nn

from app.architectures.device_packed import (
    DevicePackedEmbedding,
    DevicePackedLinear,
    upload_packed,
)
from app.architectures.layers import unpermute_rope_rows
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.dequant.torch_dequant import TorchDequantizer
from app.gguf.loader import GGUFModelLoader
from app.gguf.packed_rows import PackedRows

# Either packed token-embedding table (CPU kernels / on-device torch); both have `as_linear`.
PACKED_EMBEDDINGS = (QuantizedEmbedding, DevicePackedEmbedding)


class PackedWeightLoading:
    _quantized_loader: GGUFModelLoader | None
    # From DevicePlacement: `_device_packed` is True with `gpu_weight_mode="packed"`; supported
    # quant types are then uploaded still packed (`DevicePackedLinear`).
    _device_packed: bool
    device: torch.device

    def _mark_quantized_native_used(self, loader: GGUFModelLoader) -> None:
        """Called by `_materialize_weights` (idempotent: the per-layer loop calls it per quantized
        projection) the first time it
        builds a `QuantizedLinear` from `loader`; tells `_ensure_materialized` to keep `loader`'s
        mmap open past materialization,
        since a `QuantizedLinear`'s forward reads raw bytes from it on every future call.
        """
        self._quantized_loader = loader

    def _load_projection(
        self,
        loader: GGUFModelLoader,
        tensor_name: str,
        target: nn.Linear,
        dtype: torch.dtype,
        enabled: bool,
        bias_tensor_name: str | None = None,
        rope_heads: int | None = None,
    ) -> nn.Module:
        """Shared by every architecture's `_materialize_weights` for a projection eligible for
        quantized-native compute (`Settings.enable_quantized_native_compute`).

        When `enabled` and `tensor_name`'s GGUF type has a fused kernel
        (`quantized_gemv_registry.has_gemv_kernel`), returns a `QuantizedLinear` built from the
        loader's raw bytes and discards `target` (the placeholder `nn.Linear`). Otherwise (disabled,
        or no kernel, e.g. F16/F32) it takes the `.copy_` path and returns `target`.
        `bias_tensor_name` loads a bias either way (a `QuantizedLinear`'s bias is a small
        dequantized tensor; only the weight stays packed).

        `rope_heads` applies `unpermute_rope_rows(weight, rope_heads)` (attn_q/attn_k of
        architectures needing it). It is a pure row permutation, so the packed path reorders whole
        packed rows (a copy of just that tensor, still quantized).
        """
        if enabled and self._device_packed:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if TorchDequantizer.supports(ggml_type):
                out_features, in_features = shape
                if rope_heads is not None:
                    order = unpermute_rope_rows(torch.arange(out_features), rope_heads)
                    raw = PackedRows(raw, out_features).reordered(order)
                bias = loader.load_tensor(bias_tensor_name) if bias_tensor_name else None
                packed = upload_packed(raw, out_features, self.device)
                return DevicePackedLinear(packed, in_features, ggml_type, bias, dtype)
        elif enabled:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if has_gemv_kernel(ggml_type):
                self._mark_quantized_native_used(loader)
                out_features, in_features = shape
                if rope_heads is not None:
                    order = unpermute_rope_rows(torch.arange(out_features), rope_heads)
                    raw = PackedRows(raw, out_features).reordered(order)
                bias = loader.load_tensor(bias_tensor_name) if bias_tensor_name else None
                return QuantizedLinear(
                    out_features, in_features, ggml_type, raw, bias=bias, dtype=dtype
                )
        weight = loader.load_tensor(tensor_name)
        if rope_heads is not None:
            weight = unpermute_rope_rows(weight, rope_heads)
        target.weight.copy_(weight)
        if bias_tensor_name is not None:
            target.bias.copy_(loader.load_tensor(bias_tensor_name))
        return target

    def _load_token_embedding(
        self,
        loader: GGUFModelLoader,
        target: nn.Embedding,
        dtype: torch.dtype,
        enabled: bool,
        tensor_name: str = "token_embd.weight",
    ) -> nn.Module:
        """A packed `QuantizedEmbedding` when `enabled` and the table's type has a quantized kernel
        (the rule `_load_projection`
        uses, so a tied lm_head from `QuantizedEmbedding.as_linear` is supported too); otherwise
        `target` with the dequantized table copied in, as before.
        """
        if enabled and self._device_packed:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if TorchDequantizer.supports(ggml_type):
                packed = upload_packed(raw, shape[0], self.device)
                return DevicePackedEmbedding(packed, shape[1], ggml_type, dtype)
        elif enabled:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if has_gemv_kernel(ggml_type):
                self._mark_quantized_native_used(loader)
                num_embeddings, embedding_dim = shape
                return QuantizedEmbedding(num_embeddings, embedding_dim, ggml_type, raw, dtype)
        target.weight.copy_(loader.load_tensor(tensor_name))
        return target

    def _release_packed_modules(self) -> None:
        """Releases every packed module's live view into the loader's mmap; must run before that
        mmap closes (`ModelArchitecture.close`)."""
        for module in self.modules():
            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                module.release()
