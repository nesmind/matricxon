"""GEMV kernels for IQ1_S, IQ1_M - same real, honest performance profile as
`quantized_gemv_extended.py` (register-only accumulators, no heap scratch buffer). Correctness
cross-checked against each type's own already-oracle-validated `QuantStrategy.dequantize()` in
`iq1_family.py`. As much of each block's own per-sub-block bit-unpacking as possible is
precomputed once outside the njit hot loop, only the final per-lane grid lookup + accumulate
happens inside numba - same split as quantized_gemv_iq2.py/iq3.py.
"""

import numba
import numpy as np
import torch

from app.gguf.dequant.iq_grids import iq1s_grid_i8

_QK_K = 256
_IQ1S_DELTA = 0.125
_IQ1M_DELTA = 0.125
IQ1_S_TYPE_SIZE = 2 + 32 + 16  # 50
IQ1_M_TYPE_SIZE = 32 + 16 + 8  # 56


def _require_divisible(in_features: int, block_size: int) -> int:
    if in_features % block_size != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={block_size}")
    return in_features // block_size


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq1_s_kernel(x, dl, delta, qs, qh, out_features, n_super, grid_table):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib in range(8):
                qh_val = np.int32(qh[o, sb, ib])
                dl_val = dl[o, sb, ib]
                delta_val = delta[o, sb, ib]
                base = x_base + ib * 32
                for group in range(4):
                    idx = np.int32(qs[o, sb, ib * 4 + group]) | (((qh_val >> (3 * group)) & 7) << 8)
                    lane = base + group * 8
                    for j in range(8):
                        gval = np.float32(grid_table[idx, j])
                        acc += x[lane + j] * (dl_val * (gval + delta_val))
        y[o] = acc
    return y


def qgemv_iq1_s(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ1_S_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ1_S_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    qs = np.ascontiguousarray(blocks[:, :, 2:34])
    qh = np.ascontiguousarray(blocks[:, :, 34:50]).view("<u2").reshape(out_features, n_super, 8)

    dl = d[:, :, None] * (2 * ((qh >> 12) & 7).astype(np.float32) + 1)  # (of, ns, 8)
    delta = np.where((qh & 0x8000) != 0, -_IQ1S_DELTA, _IQ1S_DELTA).astype(np.float32)

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    result = _qgemv_iq1_s_kernel(x_np, dl, delta, qs, qh, out_features, n_super, iq1s_grid_i8())
    return torch.from_numpy(result)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq1_m_kernel(x, dl1, dl2, qs, qh, out_features, n_super, grid_table):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for ib in range(8):
                i2 = ib // 2
                d1 = dl1[o, sb, i2, ib % 2]
                d2 = dl2[o, sb, i2, ib % 2]
                qh0 = np.int32(qh[o, sb, ib * 2])
                qh1 = np.int32(qh[o, sb, ib * 2 + 1])
                idx0 = np.int32(qs[o, sb, ib * 4 + 0]) | ((qh0 << 8) & 0x700)
                idx1 = np.int32(qs[o, sb, ib * 4 + 1]) | ((qh0 << 4) & 0x700)
                idx2 = np.int32(qs[o, sb, ib * 4 + 2]) | ((qh1 << 8) & 0x700)
                idx3 = np.int32(qs[o, sb, ib * 4 + 3]) | ((qh1 << 4) & 0x700)
                delta0 = -_IQ1M_DELTA if (qh0 & 0x08) != 0 else _IQ1M_DELTA
                delta1 = -_IQ1M_DELTA if (qh0 & 0x80) != 0 else _IQ1M_DELTA
                delta2 = -_IQ1M_DELTA if (qh1 & 0x08) != 0 else _IQ1M_DELTA
                delta3 = -_IQ1M_DELTA if (qh1 & 0x80) != 0 else _IQ1M_DELTA
                base = x_base + ib * 32
                for j in range(8):
                    acc += x[base + j] * (d1 * (np.float32(grid_table[idx0, j]) + delta0))
                for j in range(8):
                    acc += x[base + 8 + j] * (d1 * (np.float32(grid_table[idx1, j]) + delta1))
                for j in range(8):
                    acc += x[base + 16 + j] * (d2 * (np.float32(grid_table[idx2, j]) + delta2))
                for j in range(8):
                    acc += x[base + 24 + j] * (d2 * (np.float32(grid_table[idx3, j]) + delta3))
        y[o] = acc
    return y


def qgemv_iq1_m(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ1_M_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ1_M_TYPE_SIZE
    )
    qs = np.ascontiguousarray(blocks[:, :, 0:32])
    qh = np.ascontiguousarray(blocks[:, :, 32:48])
    sc = np.ascontiguousarray(blocks[:, :, 48:56]).view("<u2").reshape(out_features, n_super, 4)

    scale_u16 = (
        (sc[:, :, 0] >> 12)
        | ((sc[:, :, 1] >> 8) & 0xF0)
        | ((sc[:, :, 2] >> 4) & 0xF00)
        | (sc[:, :, 3] & 0xF000)
    ).astype("<u2")
    d = scale_u16.view("<f2").astype(np.float32)  # (of, ns)

    # (of, ns, 4, 2): [.., i2, ib%2] - dl1/dl2 for the 2 "ib"s sharing sc[i2].
    dl1 = np.empty((out_features, n_super, 4, 2), dtype=np.float32)
    dl2 = np.empty((out_features, n_super, 4, 2), dtype=np.float32)
    for parity in range(2):
        shift0 = 6 * parity + 0
        shift1 = 6 * parity + 3
        dl1[:, :, :, parity] = d[:, :, None] * (2 * ((sc >> shift0) & 0x7).astype(np.float32) + 1)
        dl2[:, :, :, parity] = d[:, :, None] * (2 * ((sc >> shift1) & 0x7).astype(np.float32) + 1)

    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    result = _qgemv_iq1_m_kernel(x_np, dl1, dl2, qs, qh, out_features, n_super, iq1s_grid_i8())
    return torch.from_numpy(result)
