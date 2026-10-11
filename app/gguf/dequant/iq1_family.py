"""IQ1_S, IQ1_M: the ~1.5-1.75 bit-per-weight grid-based I-quants, the lowest-bit types here.
Layouts and algorithms verified against ggml (ggml/src/ggml-common.h + ggml-quants.c, llama.cpp
`master`). Both use iq1s_grid (app.gguf.dequant.iq_grids) but correct differently from other
I-quants: instead of a per-value sign flip, each group adds one shared `+delta`/`-delta` to every
grid value (`grid[j] + delta`): IQ1_S one delta per 8-value group from a whole-block sign bit;
IQ1_M's delta from two bits per group."""

import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy
from app.gguf.dequant.iq_grids import iq1s_grid_i8

_QK_K = 256
_IQ1S_DELTA = 0.125
_IQ1M_DELTA = 0.125


class IQ1_SStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs[32]; qh[8] (u16 each, 16 bytes)} = 50 bytes.

    `qh` is a `uint16_t[QK_K/32=8]` array in ggml (8 entries, not 16; ggml's
    `static_assert(sizeof(block_iq1_s) == sizeof(ggml_half) + QK_K/8 + QK_K/16,...)` gives 50 bytes,
    not 66). The test oracle had copied the same wrong count, hiding it; both are fixed. 8
    sub-blocks of 32 elements (4 groups of 8). Each sub-block's `qh` u16 packs a 3-bit shared scale
    exponent, a sub-block sign bit (the `delta` for all 4 groups) and 3 extra grid-index bits per
    group (extending the 8-bit `qs` byte to an 11-bit index into the 2048-entry iq1s_grid).
    """

    block_size = _QK_K
    type_size = 2 + 32 + 16

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = blocks[:, 0:2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)[:, 0]
        qs = blocks[:, 2:34]
        qh = blocks[:, 34:50].copy().view("<u2").reshape(n_blocks, 8)

        grid_table = iq1s_grid_i8()

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib in range(8):
            qh_val = qh[:, ib].astype(np.int32)
            dl = d * (2 * ((qh_val >> 12) & 7).astype(np.float32) + 1)
            delta = np.where((qh_val & 0x8000) != 0, -_IQ1S_DELTA, _IQ1S_DELTA)
            base = ib * 32
            dl_col = dl.reshape(n_blocks, 1)
            delta_col = delta.reshape(n_blocks, 1)
            for group in range(4):
                idx = qs[:, ib * 4 + group].astype(np.int32) | (((qh_val >> (3 * group)) & 7) << 8)
                grid = grid_table[idx]
                lane = base + group * 8
                out[:, lane : lane + 8] = dl_col * (grid.astype(np.float32) + delta_col)
        return torch.from_numpy(out.reshape(-1))


class IQ1_MStrategy(QuantStrategy):
    """256-element super-block: {qs[32]; qh[16]; scales[8]}, no `d` field.

    The block scale is reassembled from 4 fragments of `scales`'s 4 uint16 words into a synthetic
    fp16 pattern (`(sc[0]>>12) | ((sc[1]>>8)&0xf0) | ((sc[2]>>4)&0xf00) | (sc[3]&0xf000)`); the same
    words also supply each sub-block's 3-bit dl1/dl2 exponents at lower bit positions. 8 sub-blocks
    of 32 elements (4 groups of 8, 2 per dl1/dl2 half); each group's delta comes from one bit of its
    `qh` byte (bit 3 or 7), not a whole-sub-block sign like IQ1_S.
    """

    block_size = _QK_K
    type_size = 32 + 16 + 8

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        qs = blocks[:, 0:32]
        qh = blocks[:, 32:48]
        sc = blocks[:, 48:56].copy().view("<u2").reshape(n_blocks, 4)

        scale_u16 = (
            (sc[:, 0] >> 12)
            | ((sc[:, 1] >> 8) & 0xF0)
            | ((sc[:, 2] >> 4) & 0xF00)
            | (sc[:, 3] & 0xF000)
        ).astype("<u2")
        d = scale_u16.view("<f2").astype("<f4")

        grid_table = iq1s_grid_i8()

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib in range(8):
            i2 = ib // 2
            shift0 = 6 * (ib % 2) + 0
            shift1 = 6 * (ib % 2) + 3
            dl1 = (d * (2 * ((sc[:, i2] >> shift0) & 0x7).astype(np.float32) + 1)).reshape(
                n_blocks, 1
            )
            dl2 = (d * (2 * ((sc[:, i2] >> shift1) & 0x7).astype(np.float32) + 1)).reshape(
                n_blocks, 1
            )

            qh0 = qh[:, ib * 2].astype(np.int32)
            qh1 = qh[:, ib * 2 + 1].astype(np.int32)
            idx0 = qs[:, ib * 4 + 0].astype(np.int32) | ((qh0 << 8) & 0x700)
            idx1 = qs[:, ib * 4 + 1].astype(np.int32) | ((qh0 << 4) & 0x700)
            idx2 = qs[:, ib * 4 + 2].astype(np.int32) | ((qh1 << 8) & 0x700)
            idx3 = qs[:, ib * 4 + 3].astype(np.int32) | ((qh1 << 4) & 0x700)
            delta0 = np.where((qh0 & 0x08) != 0, -_IQ1M_DELTA, _IQ1M_DELTA).reshape(n_blocks, 1)
            delta1 = np.where((qh0 & 0x80) != 0, -_IQ1M_DELTA, _IQ1M_DELTA).reshape(n_blocks, 1)
            delta2 = np.where((qh1 & 0x08) != 0, -_IQ1M_DELTA, _IQ1M_DELTA).reshape(n_blocks, 1)
            delta3 = np.where((qh1 & 0x80) != 0, -_IQ1M_DELTA, _IQ1M_DELTA).reshape(n_blocks, 1)

            base = ib * 32
            out[:, base + 0 : base + 8] = dl1 * (grid_table[idx0].astype(np.float32) + delta0)
            out[:, base + 8 : base + 16] = dl1 * (grid_table[idx1].astype(np.float32) + delta1)
            out[:, base + 16 : base + 24] = dl2 * (grid_table[idx2].astype(np.float32) + delta2)
            out[:, base + 24 : base + 32] = dl2 * (grid_table[idx3].astype(np.float32) + delta3)
        return torch.from_numpy(out.reshape(-1))
