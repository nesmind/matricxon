"""Cross-checks the IQ4_NL/IQ4_XS/TQ1_0/TQ2_0 dequant strategies (app/gguf/dequant/iq_ternary.py)
against slow, scalar, line-by-line ports of ggml's real reference dequantize_row_iq4_nl/
dequantize_row_iq4_xs/dequantize_row_tq1_0/dequantize_row_tq2_0 functions (ggml/src/ggml-quants.c,
llama.cpp `master`) - not code reuse from the shipped strategies, just a test oracle written
independently here, matching the pattern already established in test_extended_quant_kernels.py
for Q2_K/Q3_K/Q8_K.
"""

import random
import struct

import torch

from app.gguf.dequant.iq_ternary import IQ4_NLStrategy, IQ4_XSStrategy, TQ1_0Strategy, TQ2_0Strategy

_KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]
_POW3 = [1, 3, 9, 27, 81, 243]


def _f16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _ref_dequant_iq4_nl(d: float, qs: bytes) -> list[float]:
    y: list[float] = []
    for byte in qs:
        y.append(d * _KVALUES_IQ4NL[byte & 0x0F])
    for byte in qs:
        y.append(d * _KVALUES_IQ4NL[byte >> 4])
    return y


def _ref_dequant_iq4_xs(d: float, scales_h: int, scales_l: bytes, qs: bytes) -> list[float]:
    y: list[float] = []
    qs_off = 0
    for ib in range(8):
        ls = ((scales_l[ib // 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = d * (ls - 32)
        for j in range(16):
            y.append(dl * _KVALUES_IQ4NL[qs[qs_off + j] & 0x0F])
        for j in range(16):
            y.append(dl * _KVALUES_IQ4NL[qs[qs_off + j] >> 4])
        qs_off += 16
    return y


def _ref_ternary_digit(byte: int, power: int) -> int:
    q = (byte * power) & 0xFF
    return (q * 3) >> 8


def _ref_dequant_tq1_0(d: float, qs: bytes, qh: bytes) -> list[float]:
    y: list[float] = []
    for j in range(0, 32, 32):
        for n in range(5):
            for m in range(32):
                y.append((_ref_ternary_digit(qs[j + m], _POW3[n]) - 1) * d)
    for j in range(32, 48, 16):
        for n in range(5):
            for m in range(16):
                y.append((_ref_ternary_digit(qs[j + m], _POW3[n]) - 1) * d)
    for n in range(4):
        for j in range(4):
            y.append((_ref_ternary_digit(qh[j], _POW3[n]) - 1) * d)
    return y


def _ref_dequant_tq2_0(d: float, qs: bytes) -> list[float]:
    y: list[float] = []
    for j in range(0, 64, 32):
        for shift_idx in range(4):
            shift = shift_idx * 2
            for m in range(32):
                bits = (qs[j + m] >> shift) & 3
                y.append((bits - 1) * d)
    return y


class TestIQ4_NLStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(61)
        n_blocks = 4
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            qs = _random_bytes(rng, 16)
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            raw += struct.pack("<e", d) + qs
            expected.extend(_ref_dequant_iq4_nl(d, qs))

        result = IQ4_NLStrategy().dequantize(memoryview(raw), n_elements=32 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestIQ4_XSStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(62)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            scales_h = rng.randrange(0, 65536)
            scales_l = _random_bytes(rng, 4)
            qs = _random_bytes(rng, 128)
            raw += struct.pack("<e", d) + struct.pack("<H", scales_h) + scales_l + qs
            expected.extend(_ref_dequant_iq4_xs(d, scales_h, scales_l, qs))

        result = IQ4_XSStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(
            result, torch.tensor(expected, dtype=torch.float32), atol=1e-2, rtol=1e-3
        )


class TestTQ1_0Strategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(63)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            qs = _random_bytes(rng, 48)
            qh = _random_bytes(rng, 4)
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            raw += qs + qh + struct.pack("<e", d)
            expected.extend(_ref_dequant_tq1_0(d, qs, qh))

        result = TQ1_0Strategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestTQ2_0Strategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(64)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            qs = _random_bytes(rng, 64)
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            raw += qs + struct.pack("<e", d)
            expected.extend(_ref_dequant_tq2_0(d, qs))

        result = TQ2_0Strategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert IQ4_NLStrategy().type_size == 18
        assert IQ4_XSStrategy().type_size == 136
        assert TQ1_0Strategy().type_size == 54
        assert TQ2_0Strategy().type_size == 66
