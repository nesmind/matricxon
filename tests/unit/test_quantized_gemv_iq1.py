"""Correctness for the IQ1_S/IQ1_M GEMV kernels (app/gguf/dequant/quantized_gemv_iq1.py) - same
cross-check pattern as test_quantized_gemv_extended.py: agrees with each type's own
already-oracle-validated `QuantStrategy.dequantize()` (iq1_family.py, see
test_iq1_family_quant_kernels.py) on real multi-row weight matrices.
"""

import random
import struct

import torch

from app.gguf.dequant.iq1_family import IQ1_MStrategy, IQ1_SStrategy
from app.gguf.dequant.quantized_gemv_iq1 import (
    IQ1_M_TYPE_SIZE,
    IQ1_S_TYPE_SIZE,
    qgemv_iq1_m,
    qgemv_iq1_s,
)


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _random_u16s(rng: random.Random, n: int) -> bytes:
    return b"".join(struct.pack("<H", rng.randrange(0, 65536)) for _ in range(n))


def _random_row(rng: random.Random, n_blocks: int, per_block) -> bytes:
    return b"".join(per_block(rng) for _ in range(n_blocks))


def _check(strategy, kernel, raw, out_features, in_features) -> None:
    x = torch.randn(in_features, dtype=torch.float32)
    weight = strategy.dequantize(memoryview(raw), n_elements=out_features * in_features).reshape(
        out_features, in_features
    )
    expected = x @ weight.T
    result = kernel(x, memoryview(raw), out_features, in_features)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-3)


class TestQgemvIq1S:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(93)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            d = struct.pack("<e", r.uniform(0.01, 2.0))
            return d + _random_bytes(r, 32) + _random_u16s(r, 8)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ1_SStrategy(), qgemv_iq1_s, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ1_S_TYPE_SIZE == 50


class TestQgemvIq1M:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(94)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            return _random_bytes(r, 32) + _random_bytes(r, 16) + _random_u16s(r, 4)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ1_MStrategy(), qgemv_iq1_m, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ1_M_TYPE_SIZE == 56
