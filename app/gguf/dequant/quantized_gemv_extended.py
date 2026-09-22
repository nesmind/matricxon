"""GEMV kernels for the remaining 4 K-quant types not already in `quantized_gemv.py` (Q2_K, Q3_K,
Q5_K, Q8_K) - Phase 2 of the quantized-native compute plan, same real, honest performance profile
as `qgemv_q4_k`/`qgemv_q6_k` (register-only accumulators, no heap scratch buffer). Correctness
cross-checked against each type's own already-oracle-validated `QuantStrategy.dequantize()` in
`kquants.py`/`kquants_extended.py`.

Q3_K's scale unpacking (word-level bit reassembly, not a simple per-byte mask) is reused directly
from `kquants_extended._unpack_q3k_scales` rather than re-derived - that function already runs
once per real weight-loading call (this kernel calls it once per GEMV call, same cost class), it's
only the *per-element* dequant+accumulate that needs to avoid ever materializing the full matrix.
"""

import numba
import numpy as np
import torch

from app.gguf.dequant.kquants_extended import _unpack_q3k_scales

_QK_K = 256
_K_SCALE_SIZE = 12
Q2_K_TYPE_SIZE = _QK_K // 16 + _QK_K // 4 + 2 + 2  # 84
Q3_K_TYPE_SIZE = _QK_K // 8 + _QK_K // 4 + _K_SCALE_SIZE + 2  # 110
Q5_K_TYPE_SIZE = 2 + 2 + _K_SCALE_SIZE + _QK_K // 8 + _QK_K // 2  # 176
Q8_K_TYPE_SIZE = 4 + _QK_K + (_QK_K // 16) * 2  # 292


def _require_divisible(in_features: int) -> int:
    if in_features % _QK_K != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={_QK_K}")
    return in_features // _QK_K


@numba.njit(inline="always")
def _get_scale_min_k4(j: int, scales_row: np.ndarray) -> tuple[np.uint8, np.uint8]:
    """Same as quantized_gemv.py's own copy - duplicated per that module's own established
    precedent (small per-file helper, not cross-imported) rather than a new shared import."""
    if j < 4:
        return np.uint8(scales_row[j] & 0x3F), np.uint8(scales_row[j + 4] & 0x3F)
    sc = np.uint8((scales_row[j + 4] & 0x0F) | ((scales_row[j - 4] >> 6) << 4))
    m = np.uint8((scales_row[j + 4] >> 4) | ((scales_row[j] >> 6) << 4))
    return sc, m


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q2_k_kernel(x, d, dmin, scales, qs, out_features, n_super):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_a = np.float32(0.0)
        acc_b = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            dmin_val = dmin[o, sb]
            for n in range(2):
                q_base = n * 32
                for j in range(4):
                    shift = j * 2
                    for half_sel in range(2):
                        half = half_sel * 16
                        is_ = n * 8 + j * 2 + half_sel
                        sc_val = np.float32(scales[o, sb, is_] & 0x0F)
                        mn_val = np.float32(scales[o, sb, is_] >> 4)
                        dl = d_val * sc_val
                        ml = dmin_val * mn_val
                        out_base = x_base + n * 128 + j * 32 + half
                        for lane in range(16):
                            qbyte = qs[o, sb, q_base + half + lane]
                            bits = np.float32((qbyte >> shift) & 3)
                            val = dl * bits - ml
                            if half_sel == 0:
                                acc_a += x[out_base + lane] * val
                            else:
                                acc_b += x[out_base + lane] * val
        y[o] = acc_a + acc_b
    return y


def qgemv_q2_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features)
    count = out_features * n_super * Q2_K_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, Q2_K_TYPE_SIZE
    )
    scales = np.ascontiguousarray(blocks[:, :, 0 : _QK_K // 16])
    qs = np.ascontiguousarray(blocks[:, :, _QK_K // 16 : _QK_K // 16 + _QK_K // 4])
    tail = blocks[:, :, _QK_K // 16 + _QK_K // 4 :]
    d = tail[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    dmin = tail[:, :, 2:4].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q2_k_kernel(x_np, d, dmin, scales, qs, out_features, n_super))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q3_k_kernel(x, d, scales, hmask, qs, out_features, n_super):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_a = np.float32(0.0)
        acc_b = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            for n in range(2):
                q_base = n * 32
                for j in range(4):
                    shift = j * 2
                    m_val = np.uint8(1 << (n * 4 + j))
                    for half_sel in range(2):
                        half = half_sel * 16
                        is_ = n * 8 + j * 2 + half_sel
                        dl = d_val * np.float32(scales[o, sb, is_])
                        out_base = x_base + n * 128 + j * 32 + half
                        for lane in range(16):
                            qbyte = qs[o, sb, q_base + half + lane]
                            hbyte = hmask[o, sb, half + lane]
                            bits = np.float32((qbyte >> shift) & 3)
                            sign = np.float32(0.0) if (hbyte & m_val) != 0 else np.float32(4.0)
                            val = dl * (bits - sign)
                            if half_sel == 0:
                                acc_a += x[out_base + lane] * val
                            else:
                                acc_b += x[out_base + lane] * val
        y[o] = acc_a + acc_b
    return y


def qgemv_q3_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features)
    count = out_features * n_super * Q3_K_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, Q3_K_TYPE_SIZE
    )
    hmask = np.ascontiguousarray(blocks[:, :, 0 : _QK_K // 8])
    qs = np.ascontiguousarray(blocks[:, :, _QK_K // 8 : _QK_K // 8 + _QK_K // 4])
    scales_raw = blocks[:, :, _QK_K // 8 + _QK_K // 4 : _QK_K // 8 + _QK_K // 4 + _K_SCALE_SIZE]
    d_off = _QK_K // 8 + _QK_K // 4 + _K_SCALE_SIZE
    d = (
        blocks[:, :, d_off : d_off + 2]
        .copy()
        .view("<f2")
        .astype(np.float32)
        .reshape(out_features, n_super)
    )
    scales = _unpack_q3k_scales(scales_raw.reshape(out_features * n_super, _K_SCALE_SIZE)).reshape(
        out_features, n_super, 16
    )
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q3_k_kernel(x_np, d, scales, hmask, qs, out_features, n_super))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q5_k_kernel(x, d, dmin, scales, qh, ql, out_features, n_super):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            dmin_val = dmin[o, sb]
            for c in range(4):
                is_ = c * 2
                sc1, m1 = _get_scale_min_k4(is_, scales[o, sb])
                sc2, m2 = _get_scale_min_k4(is_ + 1, scales[o, sb])
                d1 = d_val * np.float32(sc1)
                mm1 = dmin_val * np.float32(m1)
                d2 = d_val * np.float32(sc2)
                mm2 = dmin_val * np.float32(m2)
                ql_base = c * 32
                out_base = x_base + c * 64
                for lane in range(32):
                    qh_byte = qh[o, sb, lane]
                    ql_byte = ql[o, sb, ql_base + lane]
                    bit_lo = np.float32((qh_byte >> (2 * c)) & 1)
                    bit_hi = np.float32((qh_byte >> (2 * c + 1)) & 1)
                    low_val = np.float32(ql_byte & 0x0F) + bit_lo * 16.0
                    high_val = np.float32(ql_byte >> 4) + bit_hi * 16.0
                    acc_lo += x[out_base + lane] * (d1 * low_val - mm1)
                    acc_hi += x[out_base + 32 + lane] * (d2 * high_val - mm2)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q5_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features)
    count = out_features * n_super * Q5_K_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, Q5_K_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    dmin = blocks[:, :, 2:4].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    scales = np.ascontiguousarray(blocks[:, :, 4 : 4 + _K_SCALE_SIZE])
    qh_off = 4 + _K_SCALE_SIZE
    qh = np.ascontiguousarray(blocks[:, :, qh_off : qh_off + _QK_K // 8])
    ql = np.ascontiguousarray(blocks[:, :, qh_off + _QK_K // 8 :])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(
        _qgemv_q5_k_kernel(x_np, d, dmin, scales, qh, ql, out_features, n_super)
    )


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q8_k_kernel(x, d, qs, out_features, n_super):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc0 = np.float32(0.0)
        acc1 = np.float32(0.0)
        for sb in range(n_super):
            d_val = d[o, sb]
            x_base = sb * _QK_K
            for i in range(0, _QK_K, 2):
                acc0 += x[x_base + i] * (d_val * np.float32(qs[o, sb, i]))
                acc1 += x[x_base + i + 1] * (d_val * np.float32(qs[o, sb, i + 1]))
        y[o] = acc0 + acc1
    return y


def qgemv_q8_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features)
    count = out_features * n_super * Q8_K_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, Q8_K_TYPE_SIZE
    )
    d = blocks[:, :, 0:4].copy().view("<f4").reshape(out_features, n_super)
    qs = np.ascontiguousarray(blocks[:, :, 4 : 4 + _QK_K].view(np.int8))
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q8_k_kernel(x_np, d, qs, out_features, n_super))
