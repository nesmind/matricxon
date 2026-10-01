"""Python face of the native quantized matmul (`mx_gemm` in app/native/src/mx_gemm.c): y = x @ W.T
for a packed GGUF weight `W`, straight from its raw mmap'd bytes - no copy, no dequantized tensor.

`QuantizedLinear` asks `NativeGemm.active()` once per tensor at construction; it's None unless
`Settings.gemv_backend == "native"` and the library actually built, so the Numba kernels stay the
fallback for every case this doesn't cover (unsupported type, `in_features` not a multiple of
256, no C compiler).
"""

import ctypes
import logging
import os
import time

import numpy as np
import torch

from app.native.library import NativeBuildError, NativeKernelLibrary

logger = logging.getLogger(__name__)


class NativeGemm:
    _active: "NativeGemm | None" = None

    def __init__(self, lib: ctypes.CDLL, n_threads: int) -> None:
        self._lib = lib
        self._n_threads = n_threads
        self._calls = 0
        self._seconds = 0.0
        self._fallback_types_logged: set[int] = set()
        lib.mx_build_info.restype = ctypes.c_char_p
        self.build_info = lib.mx_build_info().decode()
        lib.mx_supports.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.mx_supports.restype = ctypes.c_int
        lib.mx_gemm.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mx_supports_dequant_rows.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.mx_supports_dequant_rows.restype = ctypes.c_int
        lib.mx_dequant_rows.argtypes = [
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        lib.mx_dequant_rows.restype = ctypes.c_int
        lib.mx_gemm.restype = ctypes.c_int
        lib.mx_gated_delta_rule.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5
        lib.mx_gated_delta_rule.restype = ctypes.c_int

    @classmethod
    def configure(cls, backend: str, n_threads: int | None = None) -> None:
        """Called once at startup (app.main.MatricxonApp). A failed build is logged and leaves the
        Numba kernels in charge - never a startup failure."""
        cls._active = None
        if backend != "native":
            return
        try:
            lib = NativeKernelLibrary().load()
        except (NativeBuildError, OSError) as exc:
            logger.warning("native kernels unavailable, using numba instead: %s", exc)
            return
        cls._active = cls(lib, n_threads or os.cpu_count() or 1)
        logger.info(
            "native quantized kernels enabled: %d threads, %s",
            cls._active._n_threads,
            cls._active.build_info,
        )

    @classmethod
    def active(cls) -> "NativeGemm | None":
        return cls._active

    def supports(self, ggml_type: int, in_features: int) -> bool:
        supported = bool(self._lib.mx_supports(int(ggml_type), in_features))
        if not supported and int(ggml_type) not in self._fallback_types_logged:
            self._fallback_types_logged.add(int(ggml_type))
            logger.info(
                "no native kernel for GGML type %d (in_features=%d) - numba handles it",
                int(ggml_type),
                in_features,
            )
        return supported

    def supports_dequant_rows(self, ggml_type: int, row_width: int) -> bool:
        """Whether `dequant_rows` can handle this (type, row_width) pair - only the K-quant family
        with an `mx_unpack_*` function (Q3_K/Q4_K/Q5_K/Q6_K), block-aligned. `QuantizedEmbedding`
        keeps the Python `QuantStrategy.dequantize` fallback (already correct, already used before
        this existed) for everything this returns False for."""
        return bool(self._lib.mx_supports_dequant_rows(int(ggml_type), row_width))

    def dequant_rows(
        self, ggml_type: int, raw: memoryview, row_indices: torch.Tensor, row_width: int
    ) -> torch.Tensor:
        """Dequantizes `row_indices` (1D, real row numbers into `raw`) straight to float32,
        `(len(row_indices), row_width)` - never the full table. `raw` must hold whichever total
        row count the caller's own tensor really has; only the requested rows are ever read."""
        indices32 = row_indices.detach().to(torch.int32).contiguous()
        n_rows = indices32.shape[0]
        out = torch.empty((n_rows, row_width), dtype=torch.float32)
        raw_ptr = np.frombuffer(raw, dtype=np.uint8).ctypes.data
        status = self._lib.mx_dequant_rows(
            int(ggml_type),
            raw_ptr,
            indices32.data_ptr(),
            n_rows,
            row_width,
            out.data_ptr(),
            self._n_threads,
        )
        if status != 0:
            raise RuntimeError(f"mx_dequant_rows failed with status {status} for type {ggml_type}")
        return out

    def gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        out: torch.Tensor,
    ) -> None:
        """Qwen3.5's Gated DeltaNet recurrence (see qwen35_delta_kernels.py): all float32 and
        contiguous - q/k (T,H,dk), v/out (T,H,dv), g/beta (T,H); `state` (H,dk,dv) is updated in
        place."""
        n_tokens, n_heads, dk = q.shape
        status = self._lib.mx_gated_delta_rule(
            q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
            state.data_ptr(), out.data_ptr(),
            n_tokens, n_heads, dk, v.shape[2], self._n_threads,
        )  # fmt: skip
        if status != 0:
            raise RuntimeError(f"mx_gated_delta_rule failed with status {status}")

    def take_stats(self) -> tuple[int, float]:
        """(calls, seconds) spent in native kernels since the last call - chat_router logs it per
        reply. Only ever read/reset from the one generating worker thread per model."""
        stats = (self._calls, self._seconds)
        self._calls, self._seconds = 0, 0.0
        return stats

    def matmul(
        self,
        ggml_type: int,
        raw: memoryview,
        x: torch.Tensor,
        out_features: int,
        in_features: int,
    ) -> torch.Tensor:
        """`x`: `(n_tokens, in_features)`, any float dtype. Returns `(n_tokens, out_features)`
        float32. `raw` must hold `out_features` packed rows (GGUF's own row-major order)."""
        x32 = x.detach().to(torch.float32).contiguous()
        n_tokens = x32.shape[0]
        y = torch.empty((n_tokens, out_features), dtype=torch.float32)
        # np.frombuffer only wraps the (read-only) mmap view to get its address - no copy.
        weight_ptr = np.frombuffer(raw, dtype=np.uint8).ctypes.data
        started = time.perf_counter()
        status = self._lib.mx_gemm(
            int(ggml_type),
            weight_ptr,
            x32.data_ptr(),
            y.data_ptr(),
            n_tokens,
            out_features,
            in_features,
            self._n_threads,
        )
        elapsed = time.perf_counter() - started
        if status != 0:
            raise RuntimeError(f"mx_gemm failed with status {status} for type {ggml_type}")
        self._calls += 1
        self._seconds += elapsed
        logger.debug(
            "mx_gemm type=%d %dx%d tokens=%d %.2fms",
            int(ggml_type),
            out_features,
            in_features,
            n_tokens,
            elapsed * 1000,
        )
        return y
