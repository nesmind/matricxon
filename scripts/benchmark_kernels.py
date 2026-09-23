"""Kernel-only micro-benchmark: matricxon's native C quantized matmul vs the Numba GEMV kernels vs
plain PyTorch bf16, on real weight tensors read from a real GGUF file. Seconds of load per
measurement, not minutes - the full chat benchmark (scripts/benchmark_llamacpp.py) can't finish
safely on this project's laptop (see scripts/thermal_guard.py's own docstring).

Per tensor, two shapes:
- decode (1 token): native vs numba (today's quantized-native decode path) vs torch bf16 with the
  weight already dequantized (today's default path, which holds every weight in RAM as bf16);
- prefill (N tokens): native vs today's quantized-native prefill (transient full-tensor dequant +
  matmul every call) vs torch bf16 pre-dequantized.

Usage (matricxon's own venv - needs torch/numba, not llama-cpp-python):
    .venv/bin/python scripts/benchmark_kernels.py --gguf /path/to/model.gguf
"""

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numba
import torch
from thermal_guard import ThermalGuard

from app.gguf.dequant.quantized_gemv_registry import GEMV_KERNELS
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.loader import GGUFModelLoader
from app.native.gemm import NativeGemm
from app.native.library import NativeKernelLibrary

TENSORS = ("attn_q", "attn_v", "ffn_gate", "ffn_down")


class KernelBench:
    def __init__(
        self, loader: GGUFModelLoader, guard: ThermalGuard, threads: int, budget_s: float
    ) -> None:
        self._loader = loader
        self._guard = guard
        self._budget_s = budget_s
        self._native = NativeGemm(NativeKernelLibrary().load(), threads)
        torch.set_num_threads(threads)
        numba.set_num_threads(threads)

    def _time(self, fn: Callable[[], object]) -> float:
        """Median ms per call over `budget_s`, after 2 warm-up calls (Numba JIT, caches)."""
        self._guard.wait_until_cool()
        stop = [False]
        with self._guard.watch(lambda: stop.__setitem__(0, True)):
            for _ in range(2):
                fn()
            samples = []
            deadline = time.monotonic() + self._budget_s
            while time.monotonic() < deadline and not stop[0]:
                started = time.perf_counter()
                fn()
                samples.append((time.perf_counter() - started) * 1000)
        return statistics.median(samples)

    def run(self, name: str, prefill_tokens: int) -> None:
        raw, ggml_type, (out_f, in_f) = self._loader.raw_tensor_bytes_and_type(name)
        type_name = (
            QuantStrategyRegistry().get(ggml_type).__class__.__name__.removesuffix("Strategy")
        )
        print(f"\n{name}  {type_name}  {out_f}x{in_f}")
        if not self._native.supports(ggml_type, in_f):
            print("  (no native kernel for this type - skipped)")
            return
        strategy = QuantStrategyRegistry().get(ggml_type)
        weight_bf16 = strategy.dequantize(raw, out_f * in_f).reshape(out_f, in_f).bfloat16()
        gemv, _ = GEMV_KERNELS[ggml_type]
        x1 = torch.randn(1, in_f)
        xn = torch.randn(prefill_tokens, in_f)

        def transient_dequant_matmul() -> torch.Tensor:
            weight = strategy.dequantize(raw, out_f * in_f).reshape(out_f, in_f).bfloat16()
            return xn.bfloat16() @ weight.T

        rows = {
            "decode (1 token)": {
                "native C": lambda: self._native.matmul(ggml_type, raw, x1, out_f, in_f),
                "numba": lambda: gemv(x1.reshape(-1), raw, out_f, in_f),
                "torch bf16": lambda: x1.bfloat16() @ weight_bf16.T,
            },
            f"prefill ({prefill_tokens} tokens)": {
                "native C": lambda: self._native.matmul(ggml_type, raw, xn, out_f, in_f),
                "dequant+matmul": transient_dequant_matmul,
                "torch bf16": lambda: xn.bfloat16() @ weight_bf16.T,
            },
        }
        try:
            for shape, impls in rows.items():
                native_ms = None
                for label, fn in impls.items():
                    ms = self._time(fn)  # printed right away - an overheat abort keeps the rest
                    native_ms = native_ms or ms
                    print(f"  {shape:<20} {label:<15} {ms:9.2f} ms  ({ms / native_ms:5.1f}x)")
        finally:
            # `raw` is a live view into the loader's mmap - release it explicitly, or the loader's
            # own close() fails with BufferError.
            raw.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--budget", type=float, default=0.5, help="seconds per measurement")
    parser.add_argument("--max-temp", type=float, default=90.0)
    parser.add_argument("--resume-temp", type=float, default=83.0)
    args = parser.parse_args()

    guard = ThermalGuard(max_c=args.max_temp, resume_c=args.resume_temp)
    print(f"{args.gguf.name}, layer {args.layer}, {args.threads} threads; (Nx) = time vs native C")
    with GGUFModelLoader(args.gguf) as loader:
        bench = KernelBench(loader, guard, args.threads, args.budget)
        for short in TENSORS:
            bench.run(f"blk.{args.layer}.{short}.weight", args.prefill_tokens)


if __name__ == "__main__":
    main()
