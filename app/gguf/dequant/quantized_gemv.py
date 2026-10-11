"""Quantized-native GEMV kernels. Profiling showed steady-state decode spends ~96% in
`torch._C._nn.linear` on weights matricxon dequantizes to bf16/float32 first, unlike llama.cpp,
which computes on packed blocks. Decode is always a GEMV (one activation vector, seq_len=1), a
simpler kernel than GEMM.

`qgemv_q4_k`/`qgemv_q6_k` (the K-quant types of the `ministral-3:3b` Q4_K_M file; the other 9 packed
types are in `kquants_extended.py`/`legacy.py`) compute `y = x @ W.T` for a Q4_K/Q6_K weight `W`
(`(out_features, in_features)`, GGUF's per-row block layout, as `Q4_KStrategy`/`Q6_KStrategy` in
kquants.py) by fusing dequantization into the accumulation of a Numba-JIT'd inner loop, so no
dequantized array is allocated. Parallelized with `numba.prange` over `out_features`.

Design history: the first version mixed integer nibble-unpacking and float32 multiply-accumulate in
one serial `acc +=` chain per lane and ran 1.7-3.2x slower than bf16 `nn.Linear`, since a float op
depending on a data-dependent bit shift can't vectorize and one serial accumulator gives the
compiler no independent work. Each kernel now has two phases per 256-element superblock: unpack
(integer/bit-shift work into a small `float32[256]` scratch) then accumulate (a branch-free
multiply-add over the matching slice of `x`, split across 4 independent accumulators for
instruction-level parallelism without FMA: Sandy Bridge has AVX but no AVX2/FMA3).

The per-block math equals `Q4_KStrategy`/`Q6_KStrategy.dequantize` and is cross-checked against them
in tests/unit/test_quantized_gemv.py.
"""

import numba
import numpy as np
import torch

_QK_K = 256
_K_SCALE_SIZE = 12
Q4_K_TYPE_SIZE = 2 + 2 + _K_SCALE_SIZE + _QK_K // 2
Q6_K_TYPE_SIZE = _QK_K // 2 + _QK_K // 4 + _QK_K // 16 + 2


def _require_divisible(in_features: int) -> int:
    if in_features % _QK_K != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={_QK_K}")
    return in_features // _QK_K


@numba.njit(inline="always")
def _get_scale_min_k4(j: int, scales_row: np.ndarray) -> tuple[np.uint8, np.uint8]:
    """Same bit-packing as kquants.py's `_get_scale_min_k4`, for one block's 12-byte `scales` row
    and one of its 8 (scale, min) pairs (`j`) at a time."""
    if j < 4:
        return np.uint8(scales_row[j] & 0x3F), np.uint8(scales_row[j + 4] & 0x3F)
    sc = np.uint8((scales_row[j + 4] & 0x0F) | ((scales_row[j - 4] >> 6) << 4))
    m = np.uint8((scales_row[j + 4] >> 4) | ((scales_row[j] >> 6) << 4))
    return sc, m


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q4_k_kernel(
    x: np.ndarray,
    d: np.ndarray,
    dmin: np.ndarray,
    scales: np.ndarray,
    qs: np.ndarray,
    out_features: int,
    n_super: int,
) -> np.ndarray:
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc_lo = np.float32(0.0)
        acc_hi = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            for half in range(4):
                is_ = half * 2
                sc1, m1 = _get_scale_min_k4(is_, scales[o, sb])
                sc2, m2 = _get_scale_min_k4(is_ + 1, scales[o, sb])
                d1 = d[o, sb] * np.float32(sc1)
                mm1 = dmin[o, sb] * np.float32(m1)
                d2 = d[o, sb] * np.float32(sc2)
                mm2 = dmin[o, sb] * np.float32(m2)
                q_base = half * 32
                out_base = x_base + half * 64
                # Two independent accumulator chains (acc_lo/acc_hi) instead of one serial `acc`:
                # re-benchmarking (module docstring) found a heap-allocated unpack-then-accumulate
                # scratch regressed further (extra memory traffic, no vectorization win); this keeps
                # everything in registers, breaking only the one dependency chain that was free to
                # break.
                for lane in range(32):
                    qbyte = qs[o, sb, q_base + lane]
                    acc_lo += x[out_base + lane] * (d1 * np.float32(qbyte & 0x0F) - mm1)
                    acc_hi += x[out_base + 32 + lane] * (d2 * np.float32(qbyte >> 4) - mm2)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q4_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    """`raw` holds `out_features * (in_features // 256)` Q4_K superblocks, row-major (every block of
    one output row before the next), as `Q4_KStrategy.dequantize` expects
    (`GGUFModelLoader.load_tensor`'s reshape says why that's the on-disk order).

    Header parsing (d/dmin f16->f32, scales/qs byte slicing) stays in numpy: cheap; profiling showed
    the expensive part is the per-element nibble unpack + accumulate below. Returns a
    `(out_features)` float32 tensor: `y = x @ W.T`.
    """
    n_super = _require_divisible(in_features)
    n_blocks = out_features * n_super
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q4_K_TYPE_SIZE).reshape(
        out_features, n_super, Q4_K_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    dmin = blocks[:, :, 2:4].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    scales = np.ascontiguousarray(blocks[:, :, 4 : 4 + _K_SCALE_SIZE])
    qs = np.ascontiguousarray(blocks[:, :, 4 + _K_SCALE_SIZE :])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    y = _qgemv_q4_k_kernel(x_np, d, dmin, scales, qs, out_features, n_super)
    return torch.from_numpy(y)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_q6_k_kernel(
    x: np.ndarray,
    d: np.ndarray,
    ql: np.ndarray,
    qh: np.ndarray,
    sc: np.ndarray,
    out_features: int,
    n_super: int,
) -> np.ndarray:
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc1 = np.float32(0.0)
        acc2 = np.float32(0.0)
        acc3 = np.float32(0.0)
        acc4 = np.float32(0.0)
        for sb in range(n_super):
            d_val = d[o, sb]
            x_base = sb * _QK_K
            for half in range(2):
                ql_off = half * 64
                qh_off = half * 32
                sc_off = half * 8
                base = x_base + half * 128
                # Four independent accumulator chains (acc1-acc4, one per q1-q4 value); see
                # qgemv_q4_k's kernel comment for why register-only (no heap scratch) helped.
                for lane in range(32):
                    is_ = lane // 16
                    ql_lo = ql[o, sb, ql_off + lane]
                    ql_hi = ql[o, sb, ql_off + 32 + lane]
                    qh_byte = qh[o, sb, qh_off + lane]
                    q1 = np.float32((ql_lo & 0x0F) | (((qh_byte >> 0) & 3) << 4)) - 32.0
                    q2 = np.float32((ql_hi & 0x0F) | (((qh_byte >> 2) & 3) << 4)) - 32.0
                    q3 = np.float32((ql_lo >> 4) | (((qh_byte >> 4) & 3) << 4)) - 32.0
                    q4 = np.float32((ql_hi >> 4) | (((qh_byte >> 6) & 3) << 4)) - 32.0
                    acc1 += x[base + lane] * (d_val * np.float32(sc[o, sb, sc_off + is_]) * q1)
                    acc2 += x[base + lane + 32] * (
                        d_val * np.float32(sc[o, sb, sc_off + is_ + 2]) * q2
                    )
                    acc3 += x[base + lane + 64] * (
                        d_val * np.float32(sc[o, sb, sc_off + is_ + 4]) * q3
                    )
                    acc4 += x[base + lane + 96] * (
                        d_val * np.float32(sc[o, sb, sc_off + is_ + 6]) * q4
                    )
        y[o] = acc1 + acc2 + acc3 + acc4
    return y


def qgemv_q6_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    """Same shape/layout contract as `qgemv_q4_k`, for Q6_K blocks (`Q6_KStrategy` in kquants.py has
    the field layout)."""
    n_super = _require_divisible(in_features)
    n_blocks = out_features * n_super
    blocks = np.frombuffer(raw, dtype=np.uint8, count=n_blocks * Q6_K_TYPE_SIZE).reshape(
        out_features, n_super, Q6_K_TYPE_SIZE
    )
    ql = np.ascontiguousarray(blocks[:, :, 0:128])
    qh = np.ascontiguousarray(blocks[:, :, 128:192])
    sc = np.ascontiguousarray(blocks[:, :, 192:208].view(np.int8))
    d = blocks[:, :, 208:210].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    y = _qgemv_q6_k_kernel(x_np, d, ql, qh, sc, out_features, n_super)
    return torch.from_numpy(y)
