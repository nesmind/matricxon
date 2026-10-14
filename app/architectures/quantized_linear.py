import torch
from torch import nn

from app.gguf.dequant.quantized_gemv_registry import GEMV_KERNELS
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.native.gemm import NativeGemm


class QuantizedLinear(nn.Module):
    """Quantized-native `nn.Linear` replacement: never materializes a full dequantized
    weight, the gap `Settings.enable_quantized_native_compute` names (always dequantizing to bf16 is
    why Matricxon's per-model RAM need runs far above what the same file needs elsewhere).

    Two paths, chosen per call shape: decode (`x.shape == (1, 1, in_features)`, the dominant chat
    cost) uses the fused GEMV kernel for the tensor's `ggml_type`
    (`quantized_gemv_registry.GEMV_KERNELS`) directly on `raw`, with no dequantized tensor. Prefill
    (`seq_len > 1`) transiently dequantizes via the oracle-validated `QuantStrategy.dequantize`,
    runs `torch.matmul`, then drops it (never cached into a parameter). Decode is ~1.9-4x slower
    than bf16 `nn.Linear` per kernel (see `quantized_gemv.py`): a RAM-for-speed trade, hence a
    permanent user-selectable setting.

    `raw` must stay a live view into its GGUF mmap while this module is used; see
    `ModelArchitecture._mark_quantized_native_used` for how the mmap stays open past
    materialization.
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
        # both decode and prefill below; None otherwise (Numba/dequant paths take over).
        native = NativeGemm.active()
        self._native = native if native and native.supports(ggml_type, in_features) else None
        self._strategy = QuantStrategyRegistry().get(ggml_type)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(dtype))
        else:
            self.bias = None

    def release(self) -> None:
        """Releases this module's `memoryview` into its GGUF mmap; must run
        (`ModelArchitecture.close`) before the mmap closes: a live
        memoryview blocks `mmap.close` with `BufferError: cannot close exported pointers exist`
        (live: evicting a quantized-native
        model crashed without this). `forward` must not be called afterwards: the model is being
        evicted, not idle.
        """
        self._raw = None

    @property
    def native_ready(self) -> bool:
        """Whether this weight runs on the native kernels (so it can join a `MultiGemmPlan`)."""
        return self._native is not None and self._raw is not None

    @property
    def ggml_type(self) -> int:
        return self._ggml_type

    @property
    def raw(self) -> memoryview:
        return self._raw

    def finish(self, y: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """Native output `(tokens, out_features)` float32 -> layer dtype and shape, plus bias."""
        y = y.to(self._dtype).reshape(batch, seq_len, self.out_features)
        return y + self.bias if self.bias is not None else y

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
            return self.finish(y, batch, seq_len)
        elif batch == 1 and seq_len == 1:
            # Decode step: the fused GEMV kernel, never a full dequantized weight.
            x_flat = x.reshape(-1).to(torch.float32)
            y = self._gemv_fn(x_flat, self._raw, self.out_features, self.in_features)
            y = y.to(self._dtype).reshape(1, 1, self.out_features)
        else:
            # Prefill (or batch>1; no such caller today, handled the same way): transient dequant,
            # no permanent cache.
            weight = self._strategy.dequantize(self._raw, self.out_features * self.in_features)
            weight = weight.reshape(self.out_features, self.in_features).to(self._dtype)
            y = x.to(self._dtype) @ weight.T
            del weight
        if self.bias is not None:
            y = y + self.bias
        return y
