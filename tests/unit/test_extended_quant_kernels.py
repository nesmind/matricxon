"""Cross-checks the M10-added K-quant kernels (Q2_K, Q3_K) and the plain

Q8_K kernel against slow, scalar, line-by-line ports of ggml's reference
dequantize_row_q{2,3,8}_K functions (from the public GGUF/ggml block-layout
spec - not llama.cpp/ggml code reuse in the shipped kernels, just a test
oracle written independently here), matching the pattern already established
in test_kquant_kernels.py for Q4_K/Q5_K/Q6_K.
"""

import random
import struct

import torch

from app.gguf.dequant.kquants_extended import Q2_KStrategy, Q3_KStrategy, Q8_KStrategy

_K_SCALE_SIZE = 12


def _f16_roundtrip(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def _random_bytes(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(0, 256) for _ in range(n))


def _ref_dequant_q2_k(d: float, dmin: float, scales: bytes, qs: bytes) -> list[float]:
    y: list[float] = []
    is_, q_off = 0, 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for lane in range(16):
                y.append(dl * ((qs[q_off + lane] >> shift) & 3) - ml)

            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            for lane in range(16):
                y.append(dl * ((qs[q_off + lane + 16] >> shift) & 3) - ml)

            shift += 2
        q_off += 32
    return y


def _ref_unpack_q3k_scales(scales: bytes) -> list[int]:
    """Independent Python port of ggml's aux[4]/kmask1/kmask2 word-level trick."""
    aux = list(struct.unpack("<3I", scales))
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    tmp = aux[2]
    aux2 = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux3 = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux0 = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux1 = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    packed = b"".join(struct.pack("<I", w) for w in (aux0, aux1, aux2, aux3))
    return list(struct.unpack("<16b", packed))


def _ref_dequant_q3_k(d_all: float, hmask: bytes, qs: bytes, scales: bytes) -> list[float]:
    unpacked = _ref_unpack_q3k_scales(scales)
    y: list[float] = []
    is_, m, q_off = 0, 1, 0
    for _n in range(0, 256, 128):
        shift = 0
        for _j in range(4):
            dl = d_all * (unpacked[is_] - 32)
            is_ += 1
            for lane in range(16):
                bit = 0 if (hmask[lane] & m) else 4
                y.append(dl * (((qs[q_off + lane] >> shift) & 3) - bit))

            dl = d_all * (unpacked[is_] - 32)
            is_ += 1
            for lane in range(16):
                bit = 0 if (hmask[lane + 16] & m) else 4
                y.append(dl * (((qs[q_off + lane + 16] >> shift) & 3) - bit))

            shift += 2
            m <<= 1
        q_off += 32
    return y


class TestQ2_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(47)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            scales = _random_bytes(rng, 16)
            qs = _random_bytes(rng, 64)
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            dmin = _f16_roundtrip(rng.uniform(0.01, 1.0))
            raw += scales + qs + struct.pack("<e", d) + struct.pack("<e", dmin)
            expected.extend(_ref_dequant_q2_k(d, dmin, scales, qs))

        result = Q2_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQ3_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(48)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            hmask = _random_bytes(rng, 32)
            qs = _random_bytes(rng, 64)
            scales = _random_bytes(rng, _K_SCALE_SIZE)
            d = _f16_roundtrip(rng.uniform(0.01, 2.0))
            raw += hmask + qs + scales + struct.pack("<e", d)
            expected.extend(_ref_dequant_q3_k(d, hmask, qs, scales))

        result = Q3_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestQ8_KStrategy:
    def test_matches_scalar_reference_across_multiple_blocks(self) -> None:
        rng = random.Random(49)
        n_blocks = 3
        raw = b""
        expected: list[float] = []
        for _ in range(n_blocks):
            d = rng.uniform(0.01, 2.0)
            qs = [rng.randrange(-128, 128) for _ in range(256)]
            bsums = [rng.randrange(-32768, 32768) for _ in range(16)]
            raw += struct.pack("<f", d) + struct.pack("<256b", *qs) + struct.pack("<16h", *bsums)
            expected.extend(q * d for q in qs)

        result = Q8_KStrategy().dequantize(memoryview(raw), n_elements=256 * n_blocks)

        assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


class TestByteLength:
    def test_type_sizes_match_ggml_struct_layout(self) -> None:
        assert Q2_KStrategy().type_size == 84
        assert Q3_KStrategy().type_size == 110
        assert Q8_KStrategy().type_size == 292
