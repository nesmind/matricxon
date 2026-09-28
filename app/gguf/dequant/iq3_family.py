"""IQ3_XXS, IQ3_S - the ~3-bit grid-based I-quants. Struct layouts and algorithms verified
directly against ggml's real source (ggml/src/ggml-common.h + ggml/src/ggml-quants.c,
llama.cpp `master`), not reconstructed from memory. Grid/sign tables come from
app.gguf.dequant.iq_grids (extracted verbatim from the same source) - unlike IQ2's grids
(8 values/entry), these are uint32-packed, 4 values/entry, so each group of 8 output elements
needs two separate grid lookups (grid1/grid2) rather than one.
"""

import numpy as np
import torch

from app.gguf.dequant.base import QuantStrategy
from app.gguf.dequant.iq_grids import iq3s_grid_i8, iq3xxs_grid_i8, kmask_iq2xs, ksigns_iq2xs

_QK_K = 256


def _f16_field(blocks: np.ndarray, offset: int, n_blocks: int) -> np.ndarray:
    return blocks[:, offset : offset + 2].copy().view("<f2").astype("<f4").reshape(n_blocks, 1)


class IQ3_XXSStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs[96]}.

    `qs`'s first 64 bytes are grid indices (2 per group of 8 output elements - 4-value grid
    entries need 2 lookups to fill 8 outputs); the tail 32 bytes double as 8 packed uint32
    "aux" words, each holding a 3-bit shared scale exponent plus four 7-bit sign-selectors
    (indexing ksigns_iq2xs) - same aux-word trick as IQ2_XXS, just relocated to the tail of
    the same field instead of interleaved with the indices.
    """

    block_size = _QK_K
    type_size = 2 + 96  # 98

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = _f16_field(blocks, 0, n_blocks)
        grid_idx_bytes = blocks[:, 2:66]  # (n_blocks, 64)
        aux32 = blocks[:, 66:98].copy().view("<u4")  # (n_blocks, 8)

        grid_table = iq3xxs_grid_i8()
        signs_table = ksigns_iq2xs()
        mask = kmask_iq2xs()

        db_all = d[:, 0:1] * (0.5 + (aux32 >> 28).astype(np.float32)) * 0.5  # (n_blocks, 8)

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for ib32 in range(8):
            aux = aux32[:, ib32]
            dl = db_all[:, ib32].reshape(n_blocks, 1)
            base = ib32 * 32
            for group in range(4):
                signs_sel = (aux >> (7 * group)) & 127
                signs = signs_table[signs_sel]  # (n_blocks,)
                idx1 = grid_idx_bytes[:, ib32 * 8 + 2 * group]
                idx2 = grid_idx_bytes[:, ib32 * 8 + 2 * group + 1]
                grid1 = grid_table[idx1]  # (n_blocks, 4)
                grid2 = grid_table[idx2]  # (n_blocks, 4)
                sign_mask_lo = (signs[:, None] & mask[None, 0:4]) != 0
                sign_mask_hi = (signs[:, None] & mask[None, 4:8]) != 0
                out_base = base + group * 8
                out[:, out_base : out_base + 4] = (
                    dl * grid1.astype(np.float32) * np.where(sign_mask_lo, -1.0, 1.0)
                )
                out[:, out_base + 4 : out_base + 8] = (
                    dl * grid2.astype(np.float32) * np.where(sign_mask_hi, -1.0, 1.0)
                )
        return torch.from_numpy(out.reshape(-1))


class IQ3_SStrategy(QuantStrategy):
    """256-element super-block: {d: f16; qs[64]; qh[8]; signs[32]; scales[4]}.

    Grid indices are 8 bits from `qs` extended to 9 bits by one bit pulled from `qh`; signs are
    raw bytes (no ksigns_iq2xs lookup, like IQ2_S). Processes 2 sub-blocks (64 elements) per
    scale nibble - `scales[i]`'s low nibble covers the first sub-block, high nibble the second.
    """

    block_size = _QK_K
    type_size = 2 + 64 + 8 + 32 + 4  # 110

    def dequantize(self, raw: memoryview, n_elements: int) -> torch.Tensor:
        n_blocks = n_elements // self.block_size
        blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * self.type_size).reshape(
            n_blocks, self.type_size
        )
        d = _f16_field(blocks, 0, n_blocks)
        qs = blocks[:, 2:66]  # (n_blocks, 64)
        qh = blocks[:, 66:74]  # (n_blocks, 8)
        signs = blocks[:, 74:106]  # (n_blocks, 32)
        scales = blocks[:, 106:110]  # (n_blocks, 4)

        grid_table = iq3s_grid_i8()
        mask = kmask_iq2xs()

        out = np.empty((n_blocks, _QK_K), dtype=np.float32)
        for i2 in range(4):
            db1 = d[:, 0] * (1 + 2 * (scales[:, i2] & 0x0F).astype(np.float32))
            db2 = d[:, 0] * (1 + 2 * (scales[:, i2] >> 4).astype(np.float32))
            for half, dl, qh_col, qs_off, signs_off in (
                (0, db1, 2 * i2, i2 * 16, i2 * 8),
                (1, db2, 2 * i2 + 1, i2 * 16 + 8, i2 * 8 + 4),
            ):
                qh_byte = qh[:, qh_col].astype(np.int32)
                out_base = i2 * 64 + half * 32
                dl_col = dl.reshape(n_blocks, 1)
                for group in range(4):
                    idx1 = qs[:, qs_off + 2 * group].astype(np.int32) | (
                        (qh_byte << (8 - 2 * group)) & 256
                    )
                    idx2 = qs[:, qs_off + 2 * group + 1].astype(np.int32) | (
                        (qh_byte << (7 - 2 * group)) & 256
                    )
                    grid1 = grid_table[idx1]  # (n_blocks, 4)
                    grid2 = grid_table[idx2]  # (n_blocks, 4)
                    sign_byte = signs[:, signs_off + group]
                    sign_mask_lo = (sign_byte[:, None] & mask[None, 0:4]) != 0
                    sign_mask_hi = (sign_byte[:, None] & mask[None, 4:8]) != 0
                    lane = out_base + group * 8
                    out[:, lane : lane + 4] = (
                        dl_col * grid1.astype(np.float32) * np.where(sign_mask_lo, -1.0, 1.0)
                    )
                    out[:, lane + 4 : lane + 8] = (
                        dl_col * grid2.astype(np.float32) * np.where(sign_mask_hi, -1.0, 1.0)
                    )
        return torch.from_numpy(out.reshape(-1))
