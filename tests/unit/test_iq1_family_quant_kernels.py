"""Cross-checks IQ1_S/IQ1_M (app/gguf/dequant/iq1_family.py) against slow, scalar, line-by-line
ports of ggml's real reference dequantize_row_iq1_s/iq1_m functions (ggml/src/ggml-quants.c,
llama.cpp `master`) - not code reuse from the shipped strategies, just a test oracle written
independently here, matching test_extended_quant_kernels.py's established methodology. IQ1_S's
reference here uses the corrected 8-entry (not 16-entry) `qh` field - see IQ1_SStrategy's own
docstring for why the first version of both this file and the shipped strategy shared the same
wrong assumption.
"""

import random
import struct

import torch

from app.gguf.dequant.iq1_family import IQ1_MStrategy, IQ1_SStrategy
from app.gguf.dequant.iq_grids import iq1s_grid_i8

_GRID = iq1s_grid_i8()
_IQ1S_DELTA = 0.125
_IQ1M_DELTA = 0.125


def _f16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _ref_iq1_s(d: float, qs: bytes, qh_u16: list[int]) -> list[float]:
    y: list[float] = []
    for ib in range(8):
        qh = qh_u16[ib]
        dl = d * (2 * ((qh >> 12) & 7) + 1)
        delta = -_IQ1S_DELTA if (qh & 0x8000) else _IQ1S_DELTA
        for group in range(4):
            idx = qs[ib * 4 + group] | (((qh >> (3 * group)) & 7) << 8)
            grid = _GRID[idx]
            y.extend(dl * (grid[j] + delta) for j in range(8))
    return y


def _ref_iq1_m(qs: bytes, qh: bytes, sc: list[int]) -> tuple[list[float], float]:
    scale_u16 = (
        (sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000)
    ) & 0xFFFF
    d = struct.unpack("<e", struct.pack("<H", scale_u16))[0]
    y: list[float] = []
    for ib in range(8):
        i2 = ib // 2
        dl1 = d * (2 * ((sc[i2] >> (6 * (ib % 2) + 0)) & 0x7) + 1)
        dl2 = d * (2 * ((sc[i2] >> (6 * (ib % 2) + 3)) & 0x7) + 1)
        idx = [
            qs[ib * 4 + 0] | ((qh[ib * 2 + 0] << 8) & 0x700),
            qs[ib * 4 + 1] | ((qh[ib * 2 + 0] << 4) & 0x700),
            qs[ib * 4 + 2] | ((qh[ib * 2 + 1] << 8) & 0x700),
            qs[ib * 4 + 3] | ((qh[ib * 2 + 1] << 4) & 0x700),
        ]
        delta = [
            -_IQ1M_DELTA if (qh[ib * 2 + 0] & 0x08) else _IQ1M_DELTA,
            -_IQ1M_DELTA if (qh[ib * 2 + 0] & 0x80) else _IQ1M_DELTA,
            -_IQ1M_DELTA if (qh[ib * 2 + 1] & 0x08) else _IQ1M_DELTA,
            -_IQ1M_DELTA if (qh[ib * 2 + 1] & 0x80) else _IQ1M_DELTA,
        ]
        for group in range(2):
            grid = _GRID[idx[group]]
            y.extend(dl1 * (grid[j] + delta[group]) for j in range(8))
        for group in range(2, 4):
            grid = _GRID[idx[group]]
            y.extend(dl2 * (grid[j] + delta[group]) for j in range(8))
    return y, d


class TestIQ1_SStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(401)
        n_blocks = 4
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            qs = _random_bytes(rng, 32)
            qh_u16 = [rng.randrange(0, 65536) for _ in range(8)]
            raw += struct.pack("<e", d) + qs + b"".join(struct.pack("<H", v) for v in qh_u16)
            expected.extend(_ref_iq1_s(d, qs, qh_u16))

        result = IQ1_SStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestIQ1_MStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(402)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            qs = _random_bytes(rng, 32)
            qh = _random_bytes(rng, 16)
            sc = [rng.randrange(0, 65536) for _ in range(4)]
            raw += qs + qh + b"".join(struct.pack("<H", v) for v in sc)
            y, _d = _ref_iq1_m(qs, qh, sc)
            expected.extend(y)

        result = IQ1_MStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert IQ1_SStrategy().type_size == 50
        assert IQ1_MStrategy().type_size == 56


class TestGridTable:
    def test_iq1s_grid_has_expected_shape_and_known_entries(self) -> None:
        assert _GRID.shape == (2048, 8)
        assert list(_GRID[0]) == [-1, -1, -1, -1, -1, -1, -1, -1]  # 0xffffffffffffffff
        assert list(_GRID[-1]) == [1, 1, 1, 1, 1, 1, 1, 1]  # 0x0101010101010101
