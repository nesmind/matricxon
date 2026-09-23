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
        lib.mx_gemm.restype = ctypes.c_int

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
