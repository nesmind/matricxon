"""Correctness for the remaining K-quant GEMV kernels (app/gguf/dequant/quantized_gemv_extended.py)
- same cross-check pattern as test_quantized_gemv.py: agrees with each type's own already-oracle-
validated `QuantStrategy.dequantize()` (kquants.py/kquants_extended.py) on real multi-row weight
matrices.
"""

import random
import struct

import torch

from app.gguf.dequant.kquants import Q5_KStrategy
from app.gguf.dequant.kquants_extended import Q2_KStrategy, Q3_KStrategy, Q8_KStrategy
from app.gguf.dequant.quantized_gemv_extended import (
    Q2_K_TYPE_SIZE,
    Q3_K_TYPE_SIZE,
    Q5_K_TYPE_SIZE,
    Q8_K_TYPE_SIZE,
    qgemv_q2_k,
    qgemv_q3_k,
    qgemv_q5_k,
    qgemv_q8_k,
)

_K_SCALE_SIZE = 12


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _random_row(rng: random.Random, n_super: int, per_block) -> bytes:
    return b"".join(per_block(rng) for _ in range(n_super))


def _check(strategy, kernel, raw, out_features, in_features) -> None:
    x = torch.randn(in_features, dtype=torch.float32)
    weight = strategy.dequantize(memoryview(raw), n_elements=out_features * in_features).reshape(
        out_features, in_features
    )
    expected = x @ weight.T
    result = kernel(x, memoryview(raw), out_features, in_features)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-3)


class TestQgemvQ2K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(20)
        out_features, in_features = 5, 512
        n_super = in_features // 256

        def per_block(r: random.Random) -> bytes:
            return (
                _random_bytes(r, 16)
                + _random_bytes(r, 64)
                + struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<e", r.uniform(0.01, 1.0))
            )

        raw = _random_row(rng, out_features * n_super, per_block)
        _check(Q2_KStrategy(), qgemv_q2_k, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q2_K_TYPE_SIZE == 84


class TestQgemvQ3K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(21)
        out_features, in_features = 5, 512
        n_super = in_features // 256

        def per_block(r: random.Random) -> bytes:
            return (
                _random_bytes(r, 32)
                + _random_bytes(r, 64)
                + _random_bytes(r, _K_SCALE_SIZE)
                + struct.pack("<e", r.uniform(0.01, 2.0))
            )

        raw = _random_row(rng, out_features * n_super, per_block)
        _check(Q3_KStrategy(), qgemv_q3_k, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q3_K_TYPE_SIZE == 110


class TestQgemvQ5K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(22)
        out_features, in_features = 5, 512
        n_super = in_features // 256

        def per_block(r: random.Random) -> bytes:
            return (
                struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<e", r.uniform(0.01, 1.0))
                + _random_bytes(r, _K_SCALE_SIZE)
                + _random_bytes(r, 32)
                + _random_bytes(r, 128)
            )

        raw = _random_row(rng, out_features * n_super, per_block)
        _check(Q5_KStrategy(), qgemv_q5_k, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q5_K_TYPE_SIZE == 176


class TestQgemvQ8K:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(23)
        out_features, in_features = 5, 512
        n_super = in_features // 256

        def per_block(r: random.Random) -> bytes:
            return struct.pack("<f", r.uniform(0.01, 2.0)) + struct.pack(
                "<256b", *(r.randrange(-127, 128) for _ in range(256))
            ) + _random_bytes(r, 32)

        raw = _random_row(rng, out_features * n_super, per_block)
        _check(Q8_KStrategy(), qgemv_q8_k, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q8_K_TYPE_SIZE == 292
