"""Several packed weights that share one input (q/k/v, gate/up) as one native call
(`mx_gemm_multi` in app/native/src/mx_gemm_multi.c): the input is quantized once and all rows run in
one parallel region. Built once per group (`MultiGemmPlan.build`); weight addresses are fixed mmap
addresses, so only the input and output change per call."""

import ctypes
import time

import numpy as np
import torch

from app.native.gemm import NativeGemm


class MultiGemmPlan:
    def __init__(
        self, gemm: NativeGemm, types: list[int], ptrs: list[int], outs: list[int], in_features: int
    ) -> None:
        self._gemm = gemm
        self._fn = gemm.library.mx_gemm_multi
        self._n = len(types)
        self._types = (ctypes.c_int * self._n)(*types)
        self._ptrs = (ctypes.c_void_p * self._n)(*ptrs)
        self._outs = (ctypes.c_int * self._n)(*outs)
        self.outs = list(outs)
        self.in_features = in_features
        self._total = sum(outs)

    @classmethod
    def build(
        cls, items: list[tuple[int, memoryview, int]], in_features: int
    ) -> "MultiGemmPlan | None":
        """`items`: (ggml_type, raw, out_features) per weight. None without the native library or
        when any weight lacks a native kernel."""
        gemm = NativeGemm.active()
        if gemm is None or not hasattr(gemm.library, "mx_gemm_multi") or not 1 < len(items) <= 4:
            return None
        if not all(gemm.supports(t, in_features) for t, _, _ in items):
            return None
        lib = gemm.library
        lib.mx_gemm_multi.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
        lib.mx_gemm_multi.restype = ctypes.c_int
        ptrs = [np.frombuffer(raw, dtype=np.uint8).ctypes.data for _, raw, _ in items]
        return cls(gemm, [int(t) for t, _, _ in items], ptrs, [o for _, _, o in items], in_features)

    def run(self, x: torch.Tensor) -> list[torch.Tensor] | None:
        """`x` (n_tokens, in_features): one float32 output per weight, (n_tokens, out_i); None when
        the kernels can't take this many tokens in one pass (the caller then runs each weight)."""
        x32 = x.detach().to(torch.float32).contiguous()
        y = torch.empty((x32.shape[0], self._total), dtype=torch.float32)
        started = time.perf_counter()
        status = self._fn(
            self._n, self._types, self._ptrs, self._outs, x32.data_ptr(), y.data_ptr(),
            x32.shape[0], self.in_features, self._gemm.n_threads,
        )  # fmt: skip
        if status == -1:
            return None
        self._gemm.record(time.perf_counter() - started)
        if status != 0:
            raise RuntimeError(f"mx_gemm_multi failed with status {status}")
        return list(torch.split(y, self.outs, dim=-1))
