"""Correctness for the IQ4_NL/IQ4_XS/TQ1_0/TQ2_0 GEMV kernels
(app/gguf/dequant/quantized_gemv_iq_ternary.py) - same cross-check pattern as
test_quantized_gemv_extended.py: agrees with each type's own already-oracle-validated
`QuantStrategy.dequantize()` (iq_ternary.py, see test_iq_ternary_quant_kernels.py) on real
multi-row weight matrices.
"""

import random
import struct

import torch

from app.gguf.dequant.iq_ternary import IQ4_NLStrategy, IQ4_XSStrategy, TQ1_0Strategy, TQ2_0Strategy
from app.gguf.dequant.quantized_gemv_iq_ternary import (
    IQ4_NL_TYPE_SIZE,
    IQ4_XS_TYPE_SIZE,
    TQ1_0_TYPE_SIZE,
    TQ2_0_TYPE_SIZE,
    qgemv_iq4_nl,
    qgemv_iq4_xs,
    qgemv_tq1_0,
    qgemv_tq2_0,
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


class TestQgemvIq4Nl:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(71)
        out_features, in_features = 5, 32 * 4
        n_blocks = out_features * (in_features // 32)

        def per_block(r: random.Random) -> bytes:
            return struct.pack("<e", r.uniform(0.01, 2.0)) + _random_bytes(r, 16)

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ4_NLStrategy(), qgemv_iq4_nl, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ4_NL_TYPE_SIZE == 18


class TestQgemvIq4Xs:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(72)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            return (
                struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<H", r.randrange(0, 65536))
                + _random_bytes(r, 4)
                + _random_bytes(r, 128)
            )

        raw = _random_row(rng, n_blocks, per_block)
        _check(IQ4_XSStrategy(), qgemv_iq4_xs, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert IQ4_XS_TYPE_SIZE == 136


class TestQgemvTq1_0:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(73)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            d = struct.pack("<e", r.uniform(0.01, 2.0))
            return _random_bytes(r, 48) + _random_bytes(r, 4) + d

        raw = _random_row(rng, n_blocks, per_block)
        _check(TQ1_0Strategy(), qgemv_tq1_0, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert TQ1_0_TYPE_SIZE == 54


class TestQgemvTq2_0:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(74)
        out_features, in_features = 5, 256 * 3
        n_blocks = out_features * (in_features // 256)

        def per_block(r: random.Random) -> bytes:
            return _random_bytes(r, 64) + struct.pack("<e", r.uniform(0.01, 2.0))

        raw = _random_row(rng, n_blocks, per_block)
        _check(TQ2_0Strategy(), qgemv_tq2_0, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert TQ2_0_TYPE_SIZE == 66
