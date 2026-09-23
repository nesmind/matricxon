import torch
from torch import nn

from app.gguf.dequant.quantized_gemv_registry import GEMV_KERNELS
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.native.gemm import NativeGemm


class QuantizedLinear(nn.Module):
    """Real quantized-native `nn.Linear` replacement (see ROADMAP.md/the approved plan behind
    this file): never materializes a full dequantized `(out_features, in_features)` weight - the
    real gap `Settings.enable_quantized_native_compute`'s own docstring names directly (Matricxon
    always dequantizing to bf16 before computing is why its real per-model RAM need runs far
    above what the same file needs elsewhere).

    Two real paths, chosen per real call shape, not a flag: **decode** (`x.shape == (1, 1,
    in_features)`, the dominant cost of any real chat response - many single-token forward
    passes reusing the same weights) dispatches to the real fused GEMV kernel for this tensor's
    own `ggml_type` (`quantized_gemv_registry.GEMV_KERNELS`) - operates directly on `raw`, no
    intermediate dequantized tensor ever allocated. **Prefill** (`seq_len > 1`, once per
    conversation turn) transiently dequantizes this one tensor via its own already-fast,
    already-oracle-validated `QuantStrategy.dequantize()`, runs a real `torch.matmul`, then lets
    it drop - reusing exactly the code path `GGUFModelLoader.load_tensor` already calls today,
    the only difference being the result is never cached into a permanent parameter. Real,
    honest performance profile for the decode path: ~1.9-4x slower than today's bf16 `nn.Linear`
    per real kernel (see `quantized_gemv.py`'s own docstring for why, and the real redesign
    attempts that didn't close this further) - a real RAM-for-speed trade, not a free win, which
    is exactly why it stays a real, permanent, user-selectable choice (see
    `Settings.enable_quantized_native_compute`) rather than replacing today's path outright.

    `raw` must stay a live view into its source GGUF file's mmap for as long as this module is
    used - see `ModelArchitecture._mark_quantized_native_used`'s own docstring for how the owning
    architecture keeps that mmap open past materialization for exactly this reason.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        ggml_type: int,
        raw: memoryview,
        bias: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        if ggml_type not in GEMV_KERNELS:
            raise ValueError(f"No real quantized-native GEMV kernel registered for {ggml_type!r}")
        self.out_features = out_features
        self.in_features = in_features
        self._dtype = dtype
        self._raw = raw
        self._gemv_fn, _ = GEMV_KERNELS[ggml_type]
        self._ggml_type = ggml_type
        # Settings.gemv_backend == "native" and a C kernel exists for this type/shape: it handles
        # both decode and prefill below. None otherwise - the Numba/dequant paths stay in charge.
        native = NativeGemm.active()
        self._native = native if native and native.supports(ggml_type, in_features) else None
        self._strategy = QuantStrategyRegistry().get(ggml_type)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.bias = None

    def release(self) -> None:
        """Releases this module's live `memoryview` into its source GGUF file's mmap - must be
        called (see `ModelArchitecture.close`) before that mmap is actually closed: a live
        memoryview blocks `mmap.close()` with a real `BufferError: cannot close exported pointers
        exist` otherwise (confirmed live - real eviction/unload of a model using quantized-native
        compute crashed without this). `forward()` must never be called again after this - the
        model is being evicted/unloaded, not just idle."""
        self._raw = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if self._native is not None:
            y = self._native.matmul(
                self._ggml_type,
                self._raw,
                x.reshape(-1, self.in_features),
                self.out_features,
                self.in_features,
            )
            y = y.to(self._dtype).reshape(batch, seq_len, self.out_features)
        elif batch == 1 and seq_len == 1:
            # Real decode step - the fused GEMV kernel, never a full dequantized weight.
            x_flat = x.reshape(-1).to(torch.float32)
            y = self._gemv_fn(x_flat, self._raw, self.out_features, self.in_features)
            y = y.to(self._dtype).reshape(1, 1, self.out_features)
        else:
            # Real prefill (or a batch>1 caller, none exist in this project today, handled the
            # same honest way regardless) - transient dequant, no permanent cache.
            weight = self._strategy.dequantize(self._raw, self.out_features * self.in_features)
            weight = weight.reshape(self.out_features, self.in_features).to(self._dtype)
            y = x.to(self._dtype) @ weight.T
            del weight
        if self.bias is not None:
            y = y + self.bias
        return y
