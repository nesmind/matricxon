"""GEMV kernels for IQ3_XXS, IQ3_S - same real, honest performance profile as
`quantized_gemv_extended.py` (register-only accumulators, no heap scratch buffer). Correctness
cross-checked against each type's own already-oracle-validated `QuantStrategy.dequantize()` in
`iq3_family.py`. As much of each block's own per-sub-block bit-unpacking as possible is
precomputed once outside the njit hot loop, only the final per-lane grid lookup + accumulate
happens inside numba - same split as quantized_gemv_iq2.py.
"""

import numba
import numpy as np
import torch

from app.gguf.dequant.iq_grids import iq3s_grid_i8, iq3xxs_grid_i8, kmask_iq2xs, ksigns_iq2xs

_QK_K = 256
IQ3_XXS_TYPE_SIZE = 2 + 96  # 98
IQ3_S_TYPE_SIZE = 2 + 64 + 8 + 32 + 4  # 110


def _require_divisible(in_features: int, block_size: int) -> int:
    if in_features % block_size != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={block_size}")
    return in_features // block_size


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq3_xxs_kernel(
    x, db, grid_idx_bytes, aux32, out_features, n_super, grid_table, signs_table, mask
):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib32 in range(8):
                dl = db[o, sb, ib32]
                aux = aux32[o, sb, ib32]
                base = x_base + ib32 * 32
                for group in range(4):
                    signs = signs_table[(aux >> (7 * group)) & 127]
                    idx1 = grid_idx_bytes[o, sb, ib32 * 8 + 2 * group]
                    idx2 = grid_idx_bytes[o, sb, ib32 * 8 + 2 * group + 1]
                    lane = base + group * 8
                    for j in range(4):
                        g1 = np.float32(grid_table[idx1, j])
                        sign1 = -1.0 if (signs & mask[j]) else 1.0
                        acc += x[lane + j] * (dl * g1 * sign1)
                    for j in range(4):
                        g2 = np.float32(grid_table[idx2, j])
                        sign2 = -1.0 if (signs & mask[j + 4]) else 1.0
                        acc += x[lane + 4 + j] * (dl * g2 * sign2)
        y[o] = acc
    return y


def qgemv_iq3_xxs(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ3_XXS_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ3_XXS_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    grid_idx_bytes = np.ascontiguousarray(blocks[:, :, 2:66])
    aux32 = np.ascontiguousarray(blocks[:, :, 66:98]).view("<u4").reshape(out_features, n_super, 8)

    db = d[:, :, None] * (0.5 + (aux32 >> 28).astype(np.float32)) * 0.5  # (of, ns, 8)

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    grid_table, signs_table, mask = iq3xxs_grid_i8(), ksigns_iq2xs(), kmask_iq2xs()
    result = _qgemv_iq3_xxs_kernel(
        x_np, db, grid_idx_bytes, aux32, out_features, n_super, grid_table, signs_table, mask
    )
    return torch.from_numpy(result)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq3_s_kernel(x, db1, db2, qs, qh, signs, out_features, n_super, grid_table, mask):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for i2 in range(4):
                for half in range(2):
                    dl = db1[o, sb, i2] if half == 0 else db2[o, sb, i2]
                    qh_byte = np.int32(qh[o, sb, i2 * 2 + half])
                    qs_off = i2 * 16 + half * 8
                    signs_off = i2 * 8 + half * 4
                    out_base = x_base + i2 * 64 + half * 32
                    for group in range(4):
                        idx1 = np.int32(qs[o, sb, qs_off + 2 * group]) | (
                            (qh_byte << (8 - 2 * group)) & 256
                        )
                        idx2 = np.int32(qs[o, sb, qs_off + 2 * group + 1]) | (
                            (qh_byte << (7 - 2 * group)) & 256
                        )
                        sign_byte = signs[o, sb, signs_off + group]
                        lane = out_base + group * 8
                        for j in range(4):
                            g1 = np.float32(grid_table[idx1, j])
                            sign1 = -1.0 if (sign_byte & mask[j]) else 1.0
                            acc += x[lane + j] * (dl * g1 * sign1)
                        for j in range(4):
                            g2 = np.float32(grid_table[idx2, j])
                            sign2 = -1.0 if (sign_byte & mask[j + 4]) else 1.0
                            acc += x[lane + 4 + j] * (dl * g2 * sign2)
        y[o] = acc
    return y


def qgemv_iq3_s(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ3_S_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ3_S_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    qs = np.ascontiguousarray(blocks[:, :, 2:66])
    qh = np.ascontiguousarray(blocks[:, :, 66:74])
    signs = np.ascontiguousarray(blocks[:, :, 74:106])
    scales = blocks[:, :, 106:110]  # (of, ns, 4)

    db1 = d[:, :, None] * (1 + 2 * (scales & 0x0F).astype(np.float32))  # (of, ns, 4)
    db2 = d[:, :, None] * (1 + 2 * (scales >> 4).astype(np.float32))

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    result = _qgemv_iq3_s_kernel(
        x_np, db1, db2, qs, qh, signs, out_features, n_super, iq3s_grid_i8(), kmask_iq2xs()
    )
    return torch.from_numpy(result)
