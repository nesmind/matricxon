"""IQ4_NL, IQ4_XS (I-quant, plain nonlinear-LUT family) and TQ1_0, TQ2_0 (ternary) dequant
strategies. NOT the rest of the `IQ*` family (IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S):
those need large codebook/grid tables from ggml plus sign-bit unpacking, a much larger effort.

Struct layouts and algorithms verified against ggml's source (ggml/src/ggml-common.h +
ggml/src/ggml-quants.c, llama.cpp `master`): a wrong bit layout silently produces wrong weights.
"""

import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy

_QK_K = 256

# ggml's kvalues_iq4nl: the 16-entry nonlinear lookup both IQ4_NL and IQ4_XS map their 4-bit indices
# through (their whole "codebook", a small constant, not a per-model grid like IQ1/IQ2/IQ3).
_KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)

# Ternary (base-3) digit powers; see TQ1_0Strategy for the extraction trick.
_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint8)


class IQ4_NLStrategy(QuantStrategy):
    """32-element block: {d: f16; qs[16] (two 4-bit indices per byte)}. Each nibble indexes
    _KVALUES_IQ4NL directly; no per-block sub-scale, unlike IQ4_XS."""

    block_size = 32
    type_size = 2 + 16

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        d = blocks[:, 0:2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        qs = blocks[:, 2:18]

        lo = _KVALUES_IQ4NL[qs & 0x0F]
        hi = _KVALUES_IQ4NL[qs >> 4]
        out = np.empty((n_blocks, 32), dtype=np.float32)
        out[:, 0:16] = d * lo
        out[:, 16:32] = d * hi
        return torch.from_numpy(out.reshape(-1))


class IQ4_XSStrategy(QuantStrategy):
    """256-element super-block: {d: f16; scales_h: u16; scales_l[4]; qs[128]}.

    8 sub-blocks of 32 elements; a sub-block's 6-bit signed scale is split across `scales_l` (low 4
    bits, one nibble per sub-block) and `scales_h` (high 2 bits, 2 per sub-block across all 8: the
    u16 is exactly 8*2 bits). Same _KVALUES_IQ4NL lookup as IQ4_NL.
    """

    block_size = _QK_K
    type_size = 2 + 2 + 4 + 128

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        offset = 0
        d = blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)
        offset += 2
        scales_h = blocks[:, offset : offset + 2].copy().view("<u2").reshape(n_blocks, 1)
        offset += 2
        scales_l = blocks[:, offset : offset + 4]
        offset += 4
        qs = blocks[:, offset : offset + 128]

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib in range(8):
            ls_lo = (scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0x0F
            ls_hi = ((scales_h[:, 0] >> (2 * ib)) & 3) << 4
            ls = ls_lo.astype(np.int32) | ls_hi.astype(np.int32)
            dl = (d[:, 0] * (ls - 32).astype(np.float32)).reshape(n_blocks, 1)

            sub_qs = qs[:, ib * 16 : (ib + 1) * 16]
            lo = _KVALUES_IQ4NL[sub_qs & 0x0F]
            hi = _KVALUES_IQ4NL[sub_qs >> 4]
            base = ib * 32
            out[:, base : base + 16] = dl * lo
            out[:, base + 16 : base + 32] = dl * hi
        return torch.from_numpy(out.reshape(-1))


def _ternary_digit(byte_cols: np.ndarray, power: np.uint8) -> np.ndarray:
    """Mirrors ggml's `q = byte * pow3[n]; xi = ((uint16_t) q * 3) >> 8` base-3-digit trick:
    `byte_cols * power` must wrap mod 256 like a C `uint8_t` multiply (numpy uint8 does), then the
    `>> 8` needs the wider uint16 intermediate. Returns the digit in {0, 1, 2} as float32; callers
    subtract 1 for {-1, 0, +1}."""
    q = (byte_cols * power).astype(np.uint8)
    return (q.astype(np.uint16) * 3 >> 8).astype(np.float32)


class TQ1_0Strategy(QuantStrategy):
    """256-element super-block: {qs[48]; qh[4]; d: f16}; `d` is the *last* 2 bytes here, unlike
    every other type in this file.

    Ternary (base-3: -1/0/+1) values, 5 packed per byte via `_ternary_digit`. Output order is
    digit-major, not byte-major: all 32 bytes' digit 0, then digit 1,... (ggml's
    dequantize_row_tq1_0 has three loop nests: qs[0:32] x 5 digits = 160 values, qs[32:48] remainder
    x 5 digits = 80, qh[0:4] x 4 digits (not 5) = 16; 160+80+16 = 256 = QK_K).
    """

    block_size = _QK_K
    type_size = 48 + 4 + 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        qs = blocks[:, 0:48]
        qh = blocks[:, 48:52]
        d = blocks[:, 52:54].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        col = 0
        for n in range(5):
            out[:, col : col + 32] = (_ternary_digit(qs[:, 0:32], _POW3[n]) - 1.0) * d
            col += 32
        for n in range(5):
            out[:, col : col + 16] = (_ternary_digit(qs[:, 32:48], _POW3[n]) - 1.0) * d
            col += 16
        for n in range(4):
            out[:, col : col + 4] = (_ternary_digit(qh, _POW3[n]) - 1.0) * d
            col += 4
        return torch.from_numpy(out.reshape(-1))


class TQ2_0Strategy(QuantStrategy):
    """256-element super-block: {qs[64]; d: f16}; `d` last, as TQ1_0.

    Plain 2-bit-per-value ternary (values 0..3, minus 1 -> -1/0/+1/+2: ggml's reference genuinely
    allows +2, not a clamped {-1,0,+1}; kept to match exactly). 4 values per byte, byte-major within
    each 32-byte half.
    """

    block_size = _QK_K
    type_size = 64 + 2

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )

        qs = blocks[:, 0:64]
        d = blocks[:, 64:66].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        col = 0
        for j in range(0, 64, 32):
            block32 = qs[:, j : j + 32]
            for shift_idx in range(4):
                bits = ((block32 >> (shift_idx * 2)) & 3).astype(np.float32)
                out[:, col : col + 32] = (bits - 1.0) * d
                col += 32
        return torch.from_numpy(out.reshape(-1))
