"""Cross-checks IQ2_XXS/IQ2_XS/IQ2_S (app/gguf/dequant/iq2_family.py) against slow, scalar,
line-by-line ports of ggml's real reference dequantize_row_iq2_xxs/iq2_xs/iq2_s functions
(ggml/src/ggml-quants.c, llama.cpp `master`) - not code reuse from the shipped strategies, just
a test oracle written independently here, matching test_extended_quant_kernels.py's established
methodology. IQ2_XXS's reference here uses the corrected 8-byte (not 4-byte) per-sub-block
stride - see IQ2_XXSStrategy's own docstring for why the first version of both this file and
the shipped strategy shared the same wrong assumption.
"""

import random
import struct

import torch

from app.gguf.dequant.iq2_family import IQ2_SStrategy, IQ2_XSStrategy, IQ2_XXSStrategy
from app.gguf.dequant.iq_grids import (
    iq2s_grid_i8,
    iq2xs_grid_i8,
    iq2xxs_grid_i8,
    kmask_iq2xs,
    ksigns_iq2xs,
)

_GRID_XXS = iq2xxs_grid_i8()
_GRID_XS = iq2xs_grid_i8()
_GRID_S = iq2s_grid_i8()
_SIGNS = ksigns_iq2xs()
_MASK = kmask_iq2xs()


def _f16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _sign(byte: int, j: int) -> float:
    return -1.0 if (byte & _MASK[j]) else 1.0


def _ref_iq2_xxs(d: float, qs: bytes) -> list[float]:
    y: list[float] = []
    for ib32 in range(8):
        off = ib32 * 8
        word0 = qs[off : off + 4]
        word1 = int.from_bytes(qs[off + 4 : off + 8], "little")
        db = d * (0.5 + (word1 >> 28)) * 0.25
        for group in range(4):
            grid = _GRID_XXS[word0[group]]
            signs = _SIGNS[(word1 >> (7 * group)) & 127]
            y.extend(db * grid[j] * _sign(signs, j) for j in range(8))
    return y


def _ref_iq2_xs(d: float, qs_u16: list[int], scales: bytes) -> list[float]:
    y: list[float] = []
    for ib32 in range(8):
        db0 = d * (0.5 + (scales[ib32] & 0xF)) * 0.25
        db1 = d * (0.5 + (scales[ib32] >> 4)) * 0.25
        for group in range(4):
            qval = qs_u16[ib32 * 4 + group]
            grid = _GRID_XS[qval & 511]
            signs = _SIGNS[qval >> 9]
            dl = db0 if group < 2 else db1
            y.extend(dl * grid[j] * _sign(signs, j) for j in range(8))
    return y


def _ref_iq2_s(d: float, qs_lo: bytes, sign_bytes: bytes, qh: bytes, scales: bytes) -> list[float]:
    y: list[float] = []
    for ib32 in range(8):
        db0 = d * (0.5 + (scales[ib32] & 0xF)) * 0.25
        db1 = d * (0.5 + (scales[ib32] >> 4)) * 0.25
        for group in range(4):
            base_idx = qs_lo[ib32 * 4 + group]
            extra = (qh[ib32] << (8 - 2 * group)) & 0x300
            grid = _GRID_S[base_idx | extra]
            sign_byte = sign_bytes[ib32 * 4 + group]
            dl = db0 if group < 2 else db1
            y.extend(dl * grid[j] * _sign(sign_byte, j) for j in range(8))
    return y


class TestIQ2_XXSStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(201)
        n_blocks = 4
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            qs = _random_bytes(rng, 64)
            raw += struct.pack("<e", d) + qs
            expected.extend(_ref_iq2_xxs(d, qs))

        result = IQ2_XXSStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestIQ2_XSStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(202)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            qs_u16 = [rng.randrange(0, 65536) for _ in range(32)]
            scales = _random_bytes(rng, 8)
            raw += struct.pack("<e", d) + b"".join(struct.pack("<H", v) for v in qs_u16) + scales
            expected.extend(_ref_iq2_xs(d, qs_u16, scales))

        result = IQ2_XSStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestIQ2_SStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(203)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            qs_lo = _random_bytes(rng, 32)
            sign_bytes = _random_bytes(rng, 32)
            qh = _random_bytes(rng, 8)
            scales = _random_bytes(rng, 8)
            raw += struct.pack("<e", d) + qs_lo + sign_bytes + qh + scales
            expected.extend(_ref_iq2_s(d, qs_lo, sign_bytes, qh, scales))

        result = IQ2_SStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert IQ2_XXSStrategy().type_size == 66
        assert IQ2_XSStrategy().type_size == 74
        assert IQ2_SStrategy().type_size == 82


class TestGridTables:
    """Independent of the dequant math above - catches a grid/sign-table *extraction* bug even
    if the dequant algorithm itself is otherwise correct. Values below are copy-pasted directly
    from ggml/src/ggml-common.h, not derived from this project's own extraction script."""

    def test_iq2xxs_grid_matches_known_entries(self) -> None:
        assert _GRID_XXS.shape == (256, 8)
        assert list(_GRID_XXS[0]) == [8, 8, 8, 8, 8, 8, 8, 8]  # 0x0808080808080808

    def test_iq2xs_grid_has_expected_shape(self) -> None:
        assert iq2xs_grid_i8().shape == (512, 8)

    def test_iq2s_grid_has_expected_shape(self) -> None:
        assert iq2s_grid_i8().shape == (1024, 8)

    def test_kmask_and_ksigns_match_ggml(self) -> None:
        assert list(kmask_iq2xs()) == [1, 2, 4, 8, 16, 32, 64, 128]
        assert list(ksigns_iq2xs()[0:8]) == [0, 129, 130, 3, 132, 5, 6, 135]
        assert ksigns_iq2xs()[-1] == 255
