"""Correctness for the IQ3_XXS/IQ3_S GEMV kernels (app/gguf/dequant/quantized_gemv_iq3.py) - same
cross-check pattern as test_quantized_gemv_extended.py: agrees with each type's own
already-oracle-validated `QuantStrategy.dequantize()` (iq3_family.py, see
test_iq3_family_quant_kernels.py) on real multi-row weight matrices.
"""

import random
import struct

import torch

from app.gguf.dequant.iq3_family import IQ3_SStrategy, IQ3_XXSStrategy
from app.gguf.dequant.quantized_gemv_iq3 import (
    IQ3_S_TYPE_SIZE,
    IQ3_XXS_TYPE_SIZE,
    qgemv_iq3_s,
    qgemv_iq3_xxs,
)


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


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


class TestQgemvIq3Xxs:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(91)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            d = struct.pack("<e", r.uniform(0.01, 2.0))
            return d + _random_bytes(r, 64) + _random_bytes(r, 32)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ3_XXSStrategy(), qgemv_iq3_xxs, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ3_XXS_TYPE_SIZE == 98


class TestQgemvIq3S:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(92)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            d = struct.pack("<e", r.uniform(0.01, 2.0))
            fields = [
                _random_bytes(r, 64),
                _random_bytes(r, 8),
                _random_bytes(r, 32),
                _random_bytes(r, 4),
            ]
            return d + b"".join(fields)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ3_SStrategy(), qgemv_iq3_s, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ3_S_TYPE_SIZE == 110
