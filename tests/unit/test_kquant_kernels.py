"""Cross-checks the vectorized K-quant kernels against a slow, scalar,
line-by-line port of ggml's reference dequantize_row_q{4,5,6}_K functions
(from the public GGUF/ggml block-layout spec - not llama.cpp/ggml code reuse
in the shipped kernels themselves, just a test oracle written independently
here). Any vectorization mistake in app/gguf/dequant/kquants.py should show
up as a mismatch against this intentionally naive reference.
"""

import random
import struct

import torch

from app.gguf.dequant.kquants import Q4_KStrategy, Q5_KStrategy, Q6_KStrategy

_K_SCALE_SIZE = 12


def _f16_roundtrip(value: float) -> float:
    """Rounds through float16, matching what the kernel reads back from bytes."""
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _ref_get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
    return d, m


def _ref_dequant_q4_k(d: float, dmin: float, scales: bytes, qs: bytes) -> list[float]:
    y: list[float] = []
    q_idx, is_ = 0, 0
    for _ in range(0, 256, 64):
        sc1, m1 = _ref_get_scale_min_k4(is_ + 0, scales)
        sc2, m2 = _ref_get_scale_min_k4(is_ + 1, scales)
        d1, mm1 = d * sc1, dmin * m1
        d2, mm2 = d * sc2, dmin * m2
        for lane in range(32):
            y.append(d1 * (qs[q_idx + lane] & 0xF) - mm1)
        for lane in range(32):
            y.append(d2 * (qs[q_idx + lane] >> 4) - mm2)
        q_idx += 32
        is_ += 2
    return y


def _ref_dequant_q5_k(d: float, dmin: float, scales: bytes, qh: bytes, qs: bytes) -> list[float]:
    y: list[float] = []
    q_idx, is_, u1, u2 = 0, 0, 1, 2
    for _ in range(0, 256, 64):
        sc1, m1 = _ref_get_scale_min_k4(is_ + 0, scales)
        sc2, m2 = _ref_get_scale_min_k4(is_ + 1, scales)
        d1, mm1 = d * sc1, dmin * m1
        d2, mm2 = d * sc2, dmin * m2
        for lane in range(32):
            y.append(d1 * ((qs[q_idx + lane] & 0xF) + (16 if qh[lane] & u1 else 0)) - mm1)
        for lane in range(32):
            y.append(d2 * ((qs[q_idx + lane] >> 4) + (16 if qh[lane] & u2 else 0)) - mm2)
        q_idx += 32
        is_ += 2
        u1, u2 = (u1 << 2) & 0xFF, (u2 << 2) & 0xFF
    return y


def _ref_dequant_q6_k(d: float, ql_full: bytes, qh_full: bytes, sc_full: list[int]) -> list[float]:
    y = [0.0] * 256
    for half in range(2):
        ql = ql_full[half * 64 : (half + 1) * 64]
        qh = qh_full[half * 32 : (half + 1) * 32]
        sc = sc_full[half * 8 : (half + 1) * 8]
        base = half * 128
        for lane in range(32):
            is_ = lane // 16
            q1 = ((ql[lane + 0] & 0xF) | (((qh[lane] >> 0) & 3) << 4)) - 32
            q2 = ((ql[lane + 32] & 0xF) | (((qh[lane] >> 2) & 3) << 4)) - 32
            q3 = ((ql[lane + 0] >> 4) | (((qh[lane] >> 4) & 3) << 4)) - 32
            q4 = ((ql[lane + 32] >> 4) | (((qh[lane] >> 6) & 3) << 4)) - 32
            y[base + lane + 0] = d * sc[is_ + 0] * q1
            y[base + lane + 32] = d * sc[is_ + 2] * q2
            y[base + lane + 64] = d * sc[is_ + 4] * q3
            y[base + lane + 96] = d * sc[is_ + 6] * q4
    return y


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


class TestQ4_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(42)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            dmin = _f16_roundtrip(rng.uniform(0.01, 1.0))
            scales = _random_bytes(rng, _K_SCALE_SIZE)
            qs = _random_bytes(rng, 128)
            raw += struct.pack("<e", d) + struct.pack("<e", dmin) + scales + qs
            expected.extend(_ref_dequant_q4_k(d, dmin, scales, qs))

        result = Q4_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQ5_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(43)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            dmin = _f16_roundtrip(rng.uniform(0.01, 1.0))
            scales = _random_bytes(rng, _K_SCALE_SIZE)
            qh = _random_bytes(rng, 32)
            qs = _random_bytes(rng, 128)
            raw += struct.pack("<e", d) + struct.pack("<e", dmin) + scales + qh + qs
            expected.extend(_ref_dequant_q5_k(d, dmin, scales, qh, qs))

        result = Q5_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQ6_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(44)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            ql = _random_bytes(rng, 128)
            qh = _random_bytes(rng, 64)
            sc = [rng.randrange(-30, 31) for _ in range(16)]
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            raw += ql + qh + struct.pack("<16b", *sc) + struct.pack("<e", d)
            expected.extend(_ref_dequant_q6_k(d, ql, qh, sc))

        result = Q6_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert Q4_KStrategy().type_size == 144
        assert Q5_KStrategy().type_size == 176
        assert Q6_KStrategy().type_size == 210
