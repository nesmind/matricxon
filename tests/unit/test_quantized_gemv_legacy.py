"""Correctness for the legacy-format GEMV kernels (app/gguf/dequant/quantized_gemv_legacy.py) -
same cross-check pattern as test_quantized_gemv.py's K-quant kernels: agrees with each type's own
already-oracle-validated `QuantStrategy.dequantize()` (legacy.py) on real multi-row weight
matrices, so this kernel's own math is right too without re-deriving ggml's spec a third time.
"""

import random
import struct

import torch

from app.gguf.dequant.legacy import (
    Q4_0Strategy,
    Q4_1Strategy,
    Q5_0Strategy,
    Q5_1Strategy,
    Q8_0Strategy,
)
from app.gguf.dequant.quantized_gemv_legacy import (
    Q4_0_TYPE_SIZE,
    Q4_1_TYPE_SIZE,
    Q5_0_TYPE_SIZE,
    Q5_1_TYPE_SIZE,
    Q8_0_TYPE_SIZE,
    qgemv_q4_0,
    qgemv_q4_1,
    qgemv_q5_0,
    qgemv_q5_1,
    qgemv_q8_0,
)


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _random_row(rng: random.Random, n_blocks: int, per_block: bytes) -> bytes:
    return b"".join(per_block(rng) for _ in range(n_blocks))


def _check(strategy, kernel, raw, out_features, in_features) -> None:
    x = torch.randn(in_features, dtype=torch.float32)
    weight = strategy.dequantize(memoryview(raw), n_elements=out_features * in_features).reshape(
        out_features, in_features
    )
    expected = x @ weight.T
    result = kernel(x, memoryview(raw), out_features, in_features)
    assert torch.allclose(result, expected, atol=1e-2, rtol=1e-3)


class TestQgemvQ4_0:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(10)
        out_features, in_features = 5, 64
        n_blocks = in_features // 32

        def per_block(r: random.Random) -> bytes:
            return struct.pack("<e", r.uniform(0.01, 2.0)) + _random_bytes(r, 16)

        raw = b"".join(_random_row(rng, n_blocks, per_block) for _ in range(out_features))
        _check(Q4_0Strategy(), qgemv_q4_0, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q4_0_TYPE_SIZE == 18


class TestQgemvQ4_1:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(11)
        out_features, in_features = 5, 64
        n_blocks = in_features // 32

        def per_block(r: random.Random) -> bytes:
            return (
                struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<e", r.uniform(-1.0, 1.0))
                + _random_bytes(r, 16)
            )

        raw = b"".join(_random_row(rng, n_blocks, per_block) for _ in range(out_features))
        _check(Q4_1Strategy(), qgemv_q4_1, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q4_1_TYPE_SIZE == 20


class TestQgemvQ5_0:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(12)
        out_features, in_features = 5, 64
        n_blocks = in_features // 32

        def per_block(r: random.Random) -> bytes:
            return (
                struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<I", r.randrange(0, 2**32))
                + _random_bytes(r, 16)
            )

        raw = b"".join(_random_row(rng, n_blocks, per_block) for _ in range(out_features))
        _check(Q5_0Strategy(), qgemv_q5_0, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q5_0_TYPE_SIZE == 22


class TestQgemvQ5_1:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(13)
        out_features, in_features = 5, 64
        n_blocks = in_features // 32

        def per_block(r: random.Random) -> bytes:
            return (
                struct.pack("<e", r.uniform(0.01, 2.0))
                + struct.pack("<e", r.uniform(-1.0, 1.0))
                + struct.pack("<I", r.randrange(0, 2**32))
                + _random_bytes(r, 16)
            )

        raw = b"".join(_random_row(rng, n_blocks, per_block) for _ in range(out_features))
        _check(Q5_1Strategy(), qgemv_q5_1, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q5_1_TYPE_SIZE == 24


class TestQgemvQ8_0:
    def test_matches_dequantize_then_matmul(self) -> None:
        rng = random.Random(14)
        out_features, in_features = 5, 64
        n_blocks = in_features // 32

        def per_block(r: random.Random) -> bytes:
            return struct.pack("<e", r.uniform(0.01, 2.0)) + struct.pack(
                "<32b", *(r.randrange(-127, 128) for _ in range(32))
            )

        raw = b"".join(_random_row(rng, n_blocks, per_block) for _ in range(out_features))
        _check(Q8_0Strategy(), qgemv_q8_0, raw, out_features, in_features)

    def test_type_size_matches_ggml_struct_layout(self) -> None:
        assert Q8_0_TYPE_SIZE == 34
