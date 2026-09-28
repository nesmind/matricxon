"""Cross-checks IQ3_XXS/IQ3_S (app/gguf/dequant/iq3_family.py) against slow, scalar, line-by-line
ports of ggml's real reference dequantize_row_iq3_xxs/iq3_s functions (ggml/src/ggml-quants.c,
llama.cpp `master`) - not code reuse from the shipped strategies, just a test oracle written
independently here, matching test_extended_quant_kernels.py's established methodology.
"""

import random
import struct

import torch

from app.gguf.dequant.iq3_family import IQ3_SStrategy, IQ3_XXSStrategy
from app.gguf.dequant.iq_grids import iq3s_grid_i8, iq3xxs_grid_i8, kmask_iq2xs, ksigns_iq2xs

_GRID_XXS = iq3xxs_grid_i8()
_GRID_S = iq3s_grid_i8()
_SIGNS = ksigns_iq2xs()
_MASK = kmask_iq2xs()


def _f16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _sign(byte: int, j: int) -> float:
    return -1.0 if (byte & _MASK[j]) else 1.0


def _ref_iq3_xxs(d: float, grid_idx_bytes: bytes, aux32_bytes: bytes) -> list[float]:
    y: list[float] = []
    for ib32 in range(8):
        aux = int.from_bytes(aux32_bytes[ib32 * 4 : ib32 * 4 + 4], "little")
        db = d * (0.5 + (aux >> 28)) * 0.5
        for group in range(4):
            signs = _SIGNS[(aux >> (7 * group)) & 127]
            g1 = _GRID_XXS[grid_idx_bytes[ib32 * 8 + 2 * group]]
            g2 = _GRID_XXS[grid_idx_bytes[ib32 * 8 + 2 * group + 1]]
            y.extend(db * g1[j] * _sign(signs, j) for j in range(4))
            y.extend(db * g2[j] * _sign(signs, j + 4) for j in range(4))
    return y


def _ref_iq3_s(d: float, qs: bytes, qh: bytes, signs: bytes, scales: bytes) -> list[float]:
    y: list[float] = []
    for i2 in range(4):
        db1 = d * (1 + 2 * (scales[i2] & 0xF))
        db2 = d * (1 + 2 * (scales[i2] >> 4))
        halves = (
            (db1, qh[2 * i2], i2 * 16, i2 * 8),
            (db2, qh[2 * i2 + 1], i2 * 16 + 8, i2 * 8 + 4),
        )
        for dl, qh_byte, qs_off, signs_off in halves:
            for group in range(4):
                idx1 = qs[qs_off + 2 * group] | ((qh_byte << (8 - 2 * group)) & 256)
                idx2 = qs[qs_off + 2 * group + 1] | ((qh_byte << (7 - 2 * group)) & 256)
                g1 = _GRID_S[idx1]
                g2 = _GRID_S[idx2]
                sb = signs[signs_off + group]
                y.extend(dl * g1[j] * _sign(sb, j) for j in range(4))
                y.extend(dl * g2[j] * _sign(sb, j + 4) for j in range(4))
    return y


class TestIQ3_XXSStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(301)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            grid_idx_bytes = _random_bytes(rng, 64)
            aux32_bytes = _random_bytes(rng, 32)
            raw += struct.pack("<e", d) + grid_idx_bytes + aux32_bytes
            expected.extend(_ref_iq3_xxs(d, grid_idx_bytes, aux32_bytes))

        result = IQ3_XXSStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestIQ3_SStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(302)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            qs = _random_bytes(rng, 64)
            qh = _random_bytes(rng, 8)
            signs = _random_bytes(rng, 32)
            scales = _random_bytes(rng, 4)
            raw += struct.pack("<e", d) + qs + qh + signs + scales
            expected.extend(_ref_iq3_s(d, qs, qh, signs, scales))

        result = IQ3_SStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert IQ3_XXSStrategy().type_size == 98
        assert IQ3_SStrategy().type_size == 110


class TestGridTables:
    def test_iq3xxs_grid_has_expected_shape(self) -> None:
        assert _GRID_XXS.shape == (256, 4)
        assert list(_GRID_XXS[0]) == [4, 4, 4, 4]  # 0x04040404

    def test_iq3s_grid_has_expected_shape_and_known_entries(self) -> None:
        assert _GRID_S.shape == (512, 4)
        assert list(_GRID_S[0]) == [1, 1, 1, 1]  # 0x01010101
