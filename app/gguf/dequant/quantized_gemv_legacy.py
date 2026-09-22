"""GEMV kernels for the 5 "legacy" (non-K) quant types - Phase 2 of the quantized-native compute
plan, extending `quantized_gemv.py`'s Q4_K/Q6_K design to full coverage. Same real, honest
performance profile as those two (see that module's own docstring): a register-only accumulator
design, no heap-allocated scratch buffer (proven not to help - see the plan behind this work),
correctness cross-checked against each type's own already-oracle-validated `QuantStrategy.
dequantize()` in `legacy.py`.

Each block here is much smaller (32 elements, one scale - or scale+min for Q4_1/Q5_1) than a
K-quant superblock (256 elements, 8 sub-block scales) - no sub-block bookkeeping needed, so these
kernels are simpler than qgemv_q4_k/qgemv_q6_k despite covering 5 distinct formats.
"""

import numba
import numpy as np
import torch

_BLOCK = 32
Q4_0_TYPE_SIZE = 2 + _BLOCK // 2  # 18
Q4_1_TYPE_SIZE = 2 + 2 + _BLOCK // 2  # 20
Q5_0_TYPE_SIZE = 2 + 4 + _BLOCK // 2  # 22
Q5_1_TYPE_SIZE = 2 + 2 + 4 + _BLOCK // 2  # 24
Q8_0_TYPE_SIZE = 2 + _BLOCK  # 34


def _require_divisible(in_features: int) -> int:
    if in_features % _BLOCK != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={_BLOCK}")
    return in_features // _BLOCK


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q4_0_kernel(x, d, qs, out_features, n_blocks):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for b in range(n_blocks):
            d_val = d[o, b]
            x_base = b * _BLOCK
            for lane in range(16):
                byte = qs[o, b, lane]
                acc_lo += x[x_base + lane] * (d_val * (np.float32(byte & 0x0F) - 8.0))
                acc_hi += x[x_base + 16 + lane] * (d_val * (np.float32(byte >> 4) - 8.0))
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q4_0(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features)
    count = out_features * n_blocks * Q4_0_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, Q4_0_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 2:])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q4_0_kernel(x_np, d, qs, out_features, n_blocks))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q4_1_kernel(x, d, m, qs, out_features, n_blocks):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for b in range(n_blocks):
            d_val = d[o, b]
            m_val = m[o, b]
            x_base = b * _BLOCK
            for lane in range(16):
                byte = qs[o, b, lane]
                acc_lo += x[x_base + lane] * (d_val * np.float32(byte & 0x0F) + m_val)
                acc_hi += x[x_base + 16 + lane] * (d_val * np.float32(byte >> 4) + m_val)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q4_1(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features)
    count = out_features * n_blocks * Q4_1_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, Q4_1_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    m = blocks[:, :, 2:4].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 4:])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q4_1_kernel(x_np, d, m, qs, out_features, n_blocks))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q5_0_kernel(x, d, qh, qs, out_features, n_blocks):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for b in range(n_blocks):
            d_val = d[o, b]
            qh_val = qh[o, b]
            x_base = b * _BLOCK
            for lane in range(16):
                byte = qs[o, b, lane]
                bit_lo = np.uint32((qh_val >> lane) << 4) & np.uint32(0x10)
                bit_hi = np.uint32(qh_val >> (lane + 12)) & np.uint32(0x10)
                low = np.float32(byte & 0x0F) + np.float32(bit_lo) - 16.0
                high = np.float32(byte >> 4) + np.float32(bit_hi) - 16.0
                acc_lo += x[x_base + lane] * (d_val * low)
                acc_hi += x[x_base + 16 + lane] * (d_val * high)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q5_0(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features)
    count = out_features * n_blocks * Q5_0_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, Q5_0_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qh = blocks[:, :, 2:6].copy().view("<u4").astype(np.uint32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 6:])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q5_0_kernel(x_np, d, qh, qs, out_features, n_blocks))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q5_1_kernel(x, d, m, qh, qs, out_features, n_blocks):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for b in range(n_blocks):
            d_val = d[o, b]
            m_val = m[o, b]
            qh_val = qh[o, b]
            x_base = b * _BLOCK
            for lane in range(16):
                byte = qs[o, b, lane]
                bit_lo = np.uint32((qh_val >> lane) << 4) & np.uint32(0x10)
                bit_hi = np.uint32(qh_val >> (lane + 12)) & np.uint32(0x10)
                low = np.float32(byte & 0x0F) + np.float32(bit_lo)
                high = np.float32(byte >> 4) + np.float32(bit_hi)
                acc_lo += x[x_base + lane] * (d_val * low + m_val)
                acc_hi += x[x_base + 16 + lane] * (d_val * high + m_val)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q5_1(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features)
    count = out_features * n_blocks * Q5_1_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, Q5_1_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    m = blocks[:, :, 2:4].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qh = blocks[:, :, 4:8].copy().view("<u4").astype(np.uint32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 8:])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q5_1_kernel(x_np, d, m, qh, qs, out_features, n_blocks))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q8_0_kernel(x, d, qs, out_features, n_blocks):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc0 = np.float32(0.0)
        acc1 = np.float32(0.0)
        for b in range(n_blocks):
            d_val = d[o, b]
            x_base = b * _BLOCK
            for lane in range(0, _BLOCK, 2):
                acc0 += x[x_base + lane] * (d_val * np.float32(qs[o, b, lane]))
                acc1 += x[x_base + lane + 1] * (d_val * np.float32(qs[o, b, lane + 1]))
        y[o] = acc0 + acc1
    return y


def qgemv_q8_0(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features)
    count = out_features * n_blocks * Q8_0_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, Q8_0_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 2:].view(np.int8))
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_q8_0_kernel(x_np, d, qs, out_features, n_blocks))
