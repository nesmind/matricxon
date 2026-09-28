"""GEMV kernels for IQ2_XXS, IQ2_XS, IQ2_S - same real, honest performance profile as
`quantized_gemv_extended.py` (register-only accumulators, no heap scratch buffer). Correctness
cross-checked against each type's own already-oracle-validated `QuantStrategy.dequantize()` in
`iq2_family.py`. As much of each block's own per-sub-block bit-unpacking as possible is
precomputed once outside the njit hot loop (in plain vectorized NumPy, mirroring
`quantized_gemv_extended.py`'s own `_unpack_q3k_scales`-outside-the-kernel precedent) - only the
final per-lane grid lookup + accumulate happens inside numba.
"""

import numba
import numpy as np
import torch

from app.gguf.dequant.iq_grids import (
    iq2s_grid_i8,
    iq2xs_grid_i8,
    iq2xxs_grid_i8,
    kmask_iq2xs,
    ksigns_iq2xs,
)

_QK_K = 256
IQ2_XXS_TYPE_SIZE = 2 + 64  # 66
IQ2_XS_TYPE_SIZE = 2 + 64 + 8  # 74
IQ2_S_TYPE_SIZE = 2 + 64 + 8 + 8  # 82


def _require_divisible(in_features: int, block_size: int) -> int:
    if in_features % block_size != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={block_size}")
    return in_features // block_size


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq2_xxs_kernel(
    x, db, word0, word1, out_features, n_super, grid_table, signs_table, mask
):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib32 in range(8):
                dl = db[o, sb, ib32]
                w1 = word1[o, sb, ib32]
                base = x_base + ib32 * 32
                for group in range(4):
                    grid_idx = word0[o, sb, ib32, group]
                    signs = signs_table[(w1 >> (7 * group)) & 127]
                    lane = base + group * 8
                    for j in range(8):
                        gval = np.float32(grid_table[grid_idx, j])
                        sign = -1.0 if (signs & mask[j]) else 1.0
                        acc += x[lane + j] * (dl * gval * sign)
        y[o] = acc
    return y


def qgemv_iq2_xxs(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ2_XXS_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ2_XXS_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    qs = blocks[:, :, 2:66].reshape(out_features, n_super, 8, 8)  # (of, ns, ib32, 8)

    word0 = np.ascontiguousarray(qs[:, :, :, 0:4])  # (of, ns, 8, 4) - grid indices
    word1 = np.ascontiguousarray(qs[:, :, :, 4:8]).view("<u4")[..., 0]  # (of, ns, 8) uint32
    db = d[:, :, None] * (0.5 + (word1 >> 28).astype(np.float32)) * 0.25  # (of, ns, 8)

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    grid_table, signs_table, mask = iq2xxs_grid_i8(), ksigns_iq2xs(), kmask_iq2xs()
    result = _qgemv_iq2_xxs_kernel(
        x_np, db, word0, word1, out_features, n_super, grid_table, signs_table, mask
    )
    return torch.from_numpy(result)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq2_xs_kernel(x, db0, db1, qs, out_features, n_super, grid_table, signs_table, mask):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib32 in range(8):
                base = x_base + ib32 * 32
                for group in range(4):
                    qval = qs[o, sb, ib32 * 4 + group]
                    grid_idx = qval & 511
                    signs = signs_table[qval >> 9]
                    dl = db0[o, sb, ib32] if group < 2 else db1[o, sb, ib32]
                    lane = base + group * 8
                    for j in range(8):
                        gval = np.float32(grid_table[grid_idx, j])
                        sign = -1.0 if (signs & mask[j]) else 1.0
                        acc += x[lane + j] * (dl * gval * sign)
        y[o] = acc
    return y


def qgemv_iq2_xs(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ2_XS_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ2_XS_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    qs = np.ascontiguousarray(blocks[:, :, 2:66]).view("<u2").reshape(out_features, n_super, 32)
    scales = blocks[:, :, 66:74]  # (of, ns, 8)

    db0 = d[:, :, None] * (0.5 + (scales & 0x0F).astype(np.float32)) * 0.25  # (of, ns, 8)
    db1 = d[:, :, None] * (0.5 + (scales >> 4).astype(np.float32)) * 0.25

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    grid_table, signs_table, mask = iq2xs_grid_i8(), ksigns_iq2xs(), kmask_iq2xs()
    result = _qgemv_iq2_xs_kernel(
        x_np, db0, db1, qs, out_features, n_super, grid_table, signs_table, mask
    )
    return torch.from_numpy(result)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq2_s_kernel(
    x, db0, db1, qs_lo, sign_bytes, qh, out_features, n_super, grid_table, mask
):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib32 in range(8):
                qh_val = np.int32(qh[o, sb, ib32])
                base = x_base + ib32 * 32
                for group in range(4):
                    base_idx = np.int32(qs_lo[o, sb, ib32 * 4 + group])
                    extra = (qh_val << (8 - 2 * group)) & 0x300
                    grid_idx = base_idx | extra
                    sign_byte = sign_bytes[o, sb, ib32 * 4 + group]
                    dl = db0[o, sb, ib32] if group < 2 else db1[o, sb, ib32]
                    lane = base + group * 8
                    for j in range(8):
                        gval = np.float32(grid_table[grid_idx, j])
                        sign = -1.0 if (sign_byte & mask[j]) else 1.0
                        acc += x[lane + j] * (dl * gval * sign)
        y[o] = acc
    return y


def qgemv_iq2_s(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ2_S_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ2_S_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    qs_lo = np.ascontiguousarray(blocks[:, :, 2:34])
    sign_bytes = np.ascontiguousarray(blocks[:, :, 34:66])
    qh = np.ascontiguousarray(blocks[:, :, 66:74])
    scales = blocks[:, :, 74:82]

    db0 = d[:, :, None] * (0.5 + (scales & 0x0F).astype(np.float32)) * 0.25
    db1 = d[:, :, None] * (0.5 + (scales >> 4).astype(np.float32)) * 0.25

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    result = _qgemv_iq2_s_kernel(
        x_np, db0, db1, qs_lo, sign_bytes, qh, out_features, n_super, iq2s_grid_i8(), kmask_iq2xs()
    )
    return torch.from_numpy(result)
