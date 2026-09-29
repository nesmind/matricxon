"""Correctness for the native C row-dequantize kernel (app/native/src/mx_dequant_rows.c) -
QuantizedEmbedding's real fast path for a real embedding-table lookup (app/architectures/
quantized_embedding.py): dequantizes specific rows of a packed K-quant tensor straight to
float32, never the whole table. Cross-checked against the already oracle-validated
`*Strategy.dequantize()` per row - same approach test_native_gemm.py already uses for the matmul
kernels, one random structurally-valid block per type (field order = ggml's struct/*Strategy).
"""

import random
import shutil
import struct

import pytest
import torch

from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.native.gemm import NativeGemm
from app.native.library import NativeKernelLibrary

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")


def _rand(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def _f16(rng: random.Random, lo: float, hi: float) -> bytes:
    return struct.pack("<e", rng.uniform(lo, hi))


# Same field order as test_native_gemm.py's own _BLOCKS for these 4 types - kept as its own copy
# (see mx_dequant_rows.c's own docstring on why the C side duplicates mx_gemm.c's table too).
_BLOCKS = {
    T.Q3_K: lambda r: _rand(r, 32 + 64 + 12) + _f16(r, 0.01, 1.0),
    T.Q4_K: lambda r: _f16(r, 0.01, 1.0) + _f16(r, 0.01, 1.0) + _rand(r, 12 + 128),
    T.Q5_K: lambda r: _f16(r, 0.01, 1.0) + _f16(r, 0.01, 1.0) + _rand(r, 12 + 32 + 128),
    T.Q6_K: lambda r: (
        _rand(r, 128 + 64)
        + struct.pack("<16b", *(r.randrange(-30, 31) for _ in range(16)))
        + _f16(r, 0.01, 1.0)
    ),
}


@pytest.fixture(scope="module")
def gemm(tmp_path_factory: pytest.TempPathFactory) -> NativeGemm:
    lib = NativeKernelLibrary(build_dir=tmp_path_factory.mktemp("native_build")).load()
    return NativeGemm(lib, n_threads=2)


def _table(ggml_type: T, n_rows: int, row_width: int, seed: int) -> bytes:
    rng = random.Random(seed)
    blocks_per_row = row_width // 256
    return b"".join(_BLOCKS[ggml_type](rng) for _ in range(n_rows * blocks_per_row))


def _row_block_bytes(ggml_type: T) -> int:
    return {T.Q3_K: 110, T.Q4_K: 144, T.Q5_K: 176, T.Q6_K: 210}[ggml_type]


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
def test_supports_the_full_k_quant_unpack_family(gemm: NativeGemm, ggml_type: T) -> None:
    assert gemm.supports_dequant_rows(ggml_type, 256)
    assert gemm.supports_dequant_rows(ggml_type, 512)  # 2 blocks per row


def test_does_not_support_a_type_with_no_unpack_function(gemm: NativeGemm) -> None:
    assert not gemm.supports_dequant_rows(T.Q8_0, 256)


def test_does_not_support_a_row_width_not_a_multiple_of_256(gemm: NativeGemm) -> None:
    assert not gemm.supports_dequant_rows(T.Q4_K, 200)


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
def test_dequantized_rows_match_the_reference_strategy_one_block_per_row(
    gemm: NativeGemm, ggml_type: T
) -> None:
    row_width = 256
    n_rows = 6
    block_bytes = _row_block_bytes(ggml_type)
    raw = _table(ggml_type, n_rows, row_width, seed=42)
    row_bytes = len(raw) // n_rows
    assert row_bytes == block_bytes  # sanity: exactly one block's worth per row

    strategy = QuantStrategyRegistry().get(ggml_type)
    requested = torch.tensor([4, 0, 2, 5], dtype=torch.int64)  # out of order, not every row

    result = gemm.dequant_rows(ggml_type, memoryview(raw), requested, row_width)

    for i, row_index in enumerate(requested.tolist()):
        start = row_index * row_bytes
        expected = strategy.dequantize(memoryview(raw[start : start + row_bytes]), row_width)
        assert torch.allclose(result[i], expected, atol=1e-4), f"row {row_index} mismatch"


@pytest.mark.parametrize("ggml_type", list(_BLOCKS))
def test_dequantized_rows_match_the_reference_strategy_two_blocks_per_row(
    gemm: NativeGemm, ggml_type: T
) -> None:
    """A row spanning more than one block (see mx_dequant_rows.c's own `nb` loop) - the
    single-block case above can't catch a bug that only shows up once a row has to concatenate
    more than one block's worth of dequantized values."""
    row_width = 512
    n_rows = 3
    block_bytes = _row_block_bytes(ggml_type)
    raw = _table(ggml_type, n_rows, row_width, seed=7)
    row_bytes = len(raw) // n_rows
    assert row_bytes == 2 * block_bytes

    strategy = QuantStrategyRegistry().get(ggml_type)
    requested = torch.tensor([2, 0], dtype=torch.int64)

    result = gemm.dequant_rows(ggml_type, memoryview(raw), requested, row_width)

    for i, row_index in enumerate(requested.tolist()):
        start = row_index * row_bytes
        expected = strategy.dequantize(memoryview(raw[start : start + row_bytes]), row_width)
        assert torch.allclose(result[i], expected, atol=1e-4), f"row {row_index} mismatch"
