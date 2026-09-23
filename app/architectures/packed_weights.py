"""Loading weights that stay packed (still quantized) for the model's whole lifetime - split out of
app/architectures/base.py, which every architecture's `ModelArchitecture` inherits this from.

`_load_projection` covers per-layer projections, including attn_q/attn_k with their
`unpermute_rope_rows` row permutation (applied to the packed rows, see `PackedRows`), and
`_load_token_embedding` covers the token-embedding table (see `QuantizedEmbedding`).
"""

import torch
from torch import nn

from app.architectures.layers import unpermute_rope_rows
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.quantized_gemv_registry import has_gemv_kernel
from app.gguf.loader import GGUFModelLoader
from app.gguf.packed_rows import PackedRows


class PackedWeightLoading:
    _quantized_loader: GGUFModelLoader | None

    def _mark_quantized_native_used(self, loader: GGUFModelLoader) -> None:
        """Called once by `_materialize_weights` (idempotent - a real per-layer loop calls this
        once per quantized projection, not once total) the first time it builds a real
        `QuantizedLinear` from `loader` - tells `_ensure_materialized` to keep `loader`'s mmap
        open past materialization instead of closing it, since a `QuantizedLinear`'s own forward
        pass reads real raw bytes from it on every future call, not just this one."""
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
        """Shared by every architecture's `_materialize_weights` for a real per-layer projection
        eligible for quantized-native compute (see `Settings.enable_quantized_native_compute`'s
        own docstring) - moved here once `Mistral3TextArchitecture` (the reference implementation)
        and later `llama`/`gemma4`/`phi2` all needed the identical real decision, rather than four
        separate copies of it.

        When `enabled` and `tensor_name`'s real GGUF type has a fused kernel
        (`quantized_gemv_registry.has_gemv_kernel`), returns a real `QuantizedLinear` built
        straight from this loader's raw bytes - `target` (the placeholder `nn.Linear` built in
        `__init__`) is discarded, never touched. Otherwise (disabled, or a real type with no fused
        kernel - e.g. an F16/F32 file) falls straight through to today's exact `.copy_()` path,
        returning `target` unchanged. `bias_tensor_name`, when given, loads a real bias either way
        (a `QuantizedLinear`'s own bias is always a real, small, fully-dequantized tensor - see
        that class's own docstring for why only the main weight stays packed).

        `rope_heads`, when given, applies `unpermute_rope_rows(weight, rope_heads)` - attn_q/
        attn_k of the architectures that need it. It's a pure row permutation, so on the packed
        path it reorders whole packed rows (a copy of just that tensor, still quantized) instead
        of forcing the tensor onto the float path, as it had to before.
        """
        if enabled:
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
        """A packed `QuantizedEmbedding` when `enabled` and the table's type has a quantized
        kernel (the same rule `_load_projection` uses, so a tied lm_head built from it via
        `QuantizedEmbedding.as_linear()` is always supported too); otherwise `target` with the
        dequantized table copied in, exactly as before."""
        if enabled:
            raw, ggml_type, shape = loader.raw_tensor_bytes_and_type(tensor_name)
            if has_gemv_kernel(ggml_type):
                self._mark_quantized_native_used(loader)
                num_embeddings, embedding_dim = shape
                return QuantizedEmbedding(num_embeddings, embedding_dim, ggml_type, raw, dtype)
        target.weight.copy_(loader.load_tensor(tensor_name))
        return target

    def _release_packed_modules(self) -> None:
        """Releases every packed module's live view into the loader's mmap - must run before
        that mmap is closed (see `ModelArchitecture.close`)."""
        for module in self.modules():
            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                module.release()
