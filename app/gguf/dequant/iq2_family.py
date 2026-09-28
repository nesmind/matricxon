"""IQ2_XXS, IQ2_XS, IQ2_S - the 2-bit-ish grid-based I-quants (~2.06 to ~2.56 bits/weight).
Struct layouts and algorithms verified directly against ggml's real source
(ggml/src/ggml-common.h + ggml/src/ggml-quants.c, llama.cpp `master`), not reconstructed from
memory. Grid/sign tables come from app.gguf.dequant.iq_grids (extracted verbatim from the same
source, see that module's own docstring) - a wrong bit layout here would silently produce wrong
model weights with no error.
"""

import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy
from app.gguf.dequant.iq_grids import (
    iq2s_grid_i8,
    iq2xs_grid_i8,
    iq2xxs_grid_i8,
    kmask_iq2xs,
    ksigns_iq2xs,
)

_QK_K = 256


def _f16_field(blocks: np.ndarray, offset: int, n_blocks: int) -> np.ndarray:
    return blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)


class IQ2_XXSStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs: u16[32] (66 bytes total)}.

    `qs` is a real `uint16_t[32]` array in ggml, not a flat byte buffer - the reference C reads
    each sub-block's 8-byte pair via pointer arithmetic on that `uint16_t *` (`x[i].qs +
    4*ib32`), which advances 4*sizeof(uint16_t) = 8 bytes per sub-block, not 4 (a real bug this
    implementation had until it was cross-checked against ggml's exact struct declaration - the
    independently-written test oracle had copied the same wrong assumption, so the original
    cross-check couldn't catch it; both are fixed together). Each sub-block's second uint32 word
    packs both a 3-bit shared scale exponent (top bits) and four 7-bit sign-selectors (indexing
    ksigns_iq2xs); its first word's 4 bytes are themselves the 4 grid indices for that sub-block
    (an 8-bit index per group of 8 output elements).
    """

    block_size = _QK_K
    type_size = 2 + 64  # 66

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = _f16_field(blocks, 0, n_blocks)
        qs = blocks[:, 2:66]  # (n_blocks, 64)

        grid_table = iq2xxs_grid_i8()
        signs_table = ksigns_iq2xs()
        mask = kmask_iq2xs()

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib32 in range(8):
            # Byte offset is 8*ib32 (not 4*ib32) - see this class's own docstring.
            word0 = qs[:, ib32 * 8 : ib32 * 8 + 4]  # (n_blocks, 4) - the 4 grid indices
            word1 = qs[:, ib32 * 8 + 4 : ib32 * 8 + 8].copy().view("<u4")[:, 0]  # (n_blocks,)
            db = (d[:, 0] * (0.5 + (word1 >> 28).astype(np.float32)) * 0.25).reshape(n_blocks, 1)
            base = ib32 * 32
            for group in range(4):
                grid = grid_table[word0[:, group]]  # (n_blocks, 8) int8
                signs_sel = (word1 >> (7 * group)) & 127
                signs = signs_table[signs_sel]  # (n_blocks,) uint8
                sign_mask = (signs[:, None] & mask[None, :]) != 0
                vals = db * grid.astype(np.float32) * np.where(sign_mask, -1.0, 1.0)
                out[:, base + group * 8 : base + group * 8 + 8] = vals
        return torch.from_numpy(out.reshape(-1))


class IQ2_XSStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs[32] (u16 each, 64 bytes); scales[8]}.

    Each `qs` entry packs a 9-bit grid index (low bits) plus a 7-bit sign-selector (high bits)
    directly - no extra aux word needed, unlike IQ2_XXS. Two 4-bit sub-scales per byte, shared
    across 2 groups of 8 each (so 4 groups -> 2 distinct scales, `l < 2` vs `l >= 2`).
    """

    block_size = _QK_K
    type_size = 2 + 64 + 8  # 74

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = _f16_field(blocks, 0, n_blocks)
        qs = blocks[:, 2:66].copy().view("<u2").reshape(n_blocks, 32)  # (n_blocks, 32) uint16
        scales = blocks[:, 66:74]  # (n_blocks, 8)

        grid_table = iq2xs_grid_i8()
        signs_table = ksigns_iq2xs()
        mask = kmask_iq2xs()

        db0 = d[:, 0:1] * (0.5 + (scales & 0x0F).astype(np.float32)) * 0.25  # (n_blocks, 8)
        db1 = d[:, 0:1] * (0.5 + (scales >> 4).astype(np.float32)) * 0.25  # (n_blocks, 8)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib32 in range(8):
            base = ib32 * 32
            for group in range(4):
                qval = qs[:, ib32 * 4 + group]  # (n_blocks,) uint16
                grid_idx = (qval & 511).astype(np.int64)
                signs_sel = (qval >> 9).astype(np.int64)
                grid = grid_table[grid_idx]  # (n_blocks, 8)
                signs = signs_table[signs_sel]
                sign_mask = (signs[:, None] & mask[None, :]) != 0
                dl = (db0[:, ib32] if group < 2 else db1[:, ib32]).reshape(n_blocks, 1)
                vals = dl * grid.astype(np.float32) * np.where(sign_mask, -1.0, 1.0)
                out[:, base + group * 8 : base + group * 8 + 8] = vals
        return torch.from_numpy(out.reshape(-1))


class IQ2_SStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs[64]; qh[8]; scales[8]}.

    `qs`'s 64 bytes are logically two 32-byte halves: the first half is 8-bit grid indices
    (extended to 10 bits by 2 bits pulled from `qh`), the second half is *raw* per-lane sign
    bytes used directly (unlike IQ2_XXS/IQ2_XS, no ksigns_iq2xs lookup here).
    """

    block_size = _QK_K
    type_size = 2 + 64 + 8 + 8  # 82

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = _f16_field(blocks, 0, n_blocks)
        qs_lo = blocks[:, 2:34]  # (n_blocks, 32) - grid-index bytes
        sign_bytes = blocks[:, 34:66]  # (n_blocks, 32) - raw sign bytes
        qh = blocks[:, 66:74]  # (n_blocks, 8)
        scales = blocks[:, 74:82]  # (n_blocks, 8)

        grid_table = iq2s_grid_i8()
        mask = kmask_iq2xs()

        db0 = d[:, 0:1] * (0.5 + (scales & 0x0F).astype(np.float32)) * 0.25
        db1 = d[:, 0:1] * (0.5 + (scales >> 4).astype(np.float32)) * 0.25

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib32 in range(8):
            qh_col = qh[:, ib32].astype(np.int32)
            base = ib32 * 32
            for group in range(4):
                base_idx = qs_lo[:, ib32 * 4 + group].astype(np.int32)
                extra = (qh_col << (8 - 2 * group)) & 0x300
                grid = grid_table[base_idx | extra]  # (n_blocks, 8)
                sign_byte = sign_bytes[:, ib32 * 4 + group]
                sign_mask = (sign_byte[:, None] & mask[None, :]) != 0
                dl = (db0[:, ib32] if group < 2 else db1[:, ib32]).reshape(n_blocks, 1)
                vals = dl * grid.astype(np.float32) * np.where(sign_mask, -1.0, 1.0)
                out[:, base + group * 8 : base + group * 8 + 8] = vals
        return torch.from_numpy(out.reshape(-1))
