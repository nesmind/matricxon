"""Correctness for the Numba-JIT quantized GEMV spike (app/gguf/dequant/quantized_gemv.py, see its
own module docstring and the approved plan behind it). Cross-checks against the existing
`Q4_KStrategy`/`Q6_KStrategy` dequantize-then-matmul path - already oracle-validated against a
scalar port of ggml's real reference in tests/unit/test_kquant_kernels.py - rather than re-deriving
the same per-block math a third time: if this kernel agrees with that already-proven-correct path
on real multi-row weight matrices, its own math is right too.
"""

import random
import struct

import torch

from app.gguf.dequant.kquants import Q4_KStrategy, Q6_KStrategy
from app.gguf.dequant.quantized_gemv import Q4_K_TYPE_SIZE, Q6_K_TYPE_SIZE, qgemv_q4_k, qgemv_q6_k

_K_SCALE_SIZE = 12


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _random_q4_k_row(rng: random.Random, n_super: int) -> bytes:
    raw = b""
    for _ in range(n_super):
        d = struct.pack("<e", rng.uniform(0.01, 2.0))
        dmin = struct.pack("<e", rng.uniform(0.01, 1.0))
        scales = _random_bytes(rng, _K_SCALE_SIZE)
        qs = _random_bytes(rng, 128)
        raw += d + dmin + scales + qs
    return raw


def _random_q6_k_row(rng: random.Random, n_super: int) -> bytes:
    raw = b""
    for _ in range(n_super):
        ql = _random_bytes(rng, 128)
        qh = _random_bytes(rng, 64)
        sc = struct.pack("<16b", *(rng.randrange(-30, 31) for _ in range(16)))
        d = struct.pack("<e", rng.uniform(0.01, 2.0))
        raw += ql + qh + sc + d
    return raw


class TestQgemvQ4K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(1)
        out_features, in_features = 5, 512  # 2 superblocks/row
        n_super = in_features // 256
        raw = b"".join(_random_q4_k_row(rng, n_super) for _ in range(out_features))
        x = torch.randn(in_features, dtype=torch.float32)

        weight = (
            Q4_KStrategy()
            .dequantize(memoryview(raw), n_elements=out_features * in_features)
            .reshape(out_features, in_features)
        )
        expected = x @ weight.T

        result = qgemv_q4_k(x, memoryview(raw), out_features, in_features)

        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-3)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q4_K_TYPE_SIZE == 144


class TestQgemvQ6K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(2)
        out_features, in_features = 5, 512
        n_super = in_features // 256
        raw = b"".join(_random_q6_k_row(rng, n_super) for _ in range(out_features))
        x = torch.randn(in_features, dtype=torch.float32)

        weight = (
            Q6_KStrategy()
            .dequantize(memoryview(raw), n_elements=out_features * in_features)
            .reshape(out_features, in_features)
        )
        expected = x @ weight.T

        result = qgemv_q6_k(x, memoryview(raw), out_features, in_features)

        assert torch.allclose(result, expected, atol=1e-2, rtol=1e-3)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q6_K_TYPE_SIZE == 210
