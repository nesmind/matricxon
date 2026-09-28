"""Correctness for the IQ2_XXS/IQ2_XS/IQ2_S GEMV kernels
(app/gguf/dequant/quantized_gemv_iq2.py) - same cross-check pattern as
test_quantized_gemv_extended.py: agrees with each type's own already-oracle-validated
`QuantStrategy.dequantize()` (iq2_family.py, see test_iq2_family_quant_kernels.py) on real
multi-row weight matrices.
"""

import random
import struct

import torch

from app.gguf.dequant.iq2_family import IQ2_SStrategy, IQ2_XSStrategy, IQ2_XXSStrategy
from app.gguf.dequant.quantized_gemv_iq2 import (
    IQ2_S_TYPE_SIZE,
    IQ2_XS_TYPE_SIZE,
    IQ2_XXS_TYPE_SIZE,
    qgemv_iq2_s,
    qgemv_iq2_xs,
    qgemv_iq2_xxs,
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


class TestQgemvIq2Xxs:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(81)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            return struct.pack("<e", r.uniform(0.01, 2.0)) + _random_bytes(r, 64)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ2_XXSStrategy(), qgemv_iq2_xxs, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ2_XXS_TYPE_SIZE == 66


class TestQgemvIq2Xs:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(82)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            qs = b"".join(struct.pack("<H", r.randrange(0, 65536)) for _ in range(32))
            return struct.pack("<e", r.uniform(0.01, 2.0)) + qs + _random_bytes(r, 8)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ2_XSStrategy(), qgemv_iq2_xs, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ2_XS_TYPE_SIZE == 74


class TestQgemvIq2S:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(83)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            d = struct.pack("<e", r.uniform(0.01, 2.0))
            fields = [
                _random_bytes(r, 32),
                _random_bytes(r, 32),
                _random_bytes(r, 8),
                _random_bytes(r, 8),
            ]
            return d + b"".join(fields)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ2_SStrategy(), qgemv_iq2_s, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ2_S_TYPE_SIZE == 82
