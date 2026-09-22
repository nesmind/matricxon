"""Real quantized-native GEMV kernels (see ROADMAP.md's "Known issues" and the approved plan
behind this file): real profiling this session found matricxon's steady-state decode time ~96%
inside `torch._C._nn.linear`, on weights matricxon always dequantizes to bf16/float32 first - the
opposite of llama.cpp's own approach, which computes directly against packed quantized blocks and
never materializes a dequantized weight matrix at all. Decode is always a GEMV (single activation
vector, seq_len=1), not the general batched GEMM case - a much simpler kernel to write.

These two functions - `qgemv_q4_k`/`qgemv_q6_k`, the two K-quant types the real `ministral-3:3b`
Q4_K_M file actually uses (confirmed via its own GGUF metadata; see `app/gguf/dequant/
kquants_extended.py`/`legacy.py` for the remaining 9 packed types this same design extends to) -
compute `y = x @ W.T` for a Q4_K/Q6_K-quantized weight matrix `W` (shape
`(out_features, in_features)`, GGUF's own per-row block layout, matching
`Q4_KStrategy`/`Q6_KStrategy` in kquants.py) by fusing dequantization into the accumulation inside
a Numba-JIT'd inner loop, so no full dequantized `(out_features, in_features)` array is ever
allocated. Parallelized with `numba.prange` across `out_features` - the natural embarrassingly-
parallel axis for a GEMV, matching this project's target machine's 4 real cores.

**Phase 1 redesign (real, honest history, not hidden)**: a first version of this kernel mixed
integer nibble-unpacking and the float32 multiply-accumulate in one serial `acc +=` chain, per
lane - real benchmarking found that 1.7-3.2x *slower* than bf16 `nn.Linear`, because that shape
defeats auto-vectorization (a float op whose value depends on a same-iteration data-dependent bit
shift can't be vectorized, and one serial accumulator gives the compiler no independent work to
schedule in parallel). Each kernel below is restructured into two explicit phases per real 256-
element superblock: **unpack** (pure integer/bit-shift work into a small real `float32[256]` local
scratch buffer, no accumulation) then **accumulate** (a clean, branch-free multiply-add loop over
that scratch buffer against the matching real slice of `x`, split across 4 independent
accumulators to break the serial dependency chain and give LLVM/Numba real instruction-level
parallelism to exploit even without FMA hardware - this machine's real CPU, Sandy Bridge, has AVX
but no AVX2/FMA3, see the plan's own Context section). Real re-benchmark after this redesign:
`bench_gemv.py` (scratchpad, real Ministral-3B shapes) - see the plan/ROADMAP for the actual
numbers this produced.

The per-block math itself (unchanged by this redesign) is the same as `Q4_KStrategy`/
`Q6_KStrategy`'s own dequantize() methods and is cross-checked against them directly in
tests/unit/test_quantized_gemv.py, not re-derived from ggml's spec a third time - those two are
already oracle-validated there.
"""

import numba
import numpy as np
import torch

_QK_K = 256
_K_SCALE_SIZE = 12
Q4_K_TYPE_SIZE = 2 + 2 + _K_SCALE_SIZE + _QK_K // 2  # 144
Q6_K_TYPE_SIZE = _QK_K // 2 + _QK_K // 4 + _QK_K // 16 + 2  # 210


def _require_divisible(in_features: int) -> int:
    if in_features % _QK_K != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={_QK_K}")
    return in_features // _QK_K


@numba.njit(inline="always")
def _get_scale_min_k4(j: int, scales_row: np.ndarray) -> tuple[np.uint8, np.uint8]:
    """Same bit-packing as kquants.py's `_get_scale_min_k4`, for one block's 12-byte `scales` row
    and one of its 8 (scale, min) pairs (`j`) at a time - see that function's own docstring."""
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
                # Two independent accumulator chains (acc_lo/acc_hi) instead of one serial `acc`
                # - real re-benchmarking (see this module's own docstring) found a heap-allocated
                # unpack-then-accumulate scratch buffer regressed further (extra memory traffic
                # with no compensating vectorization win); this keeps everything in registers,
                # only breaking the one dependency chain that was actually free to break.
                for lane in range(32):
                    qbyte = qs[o, sb, q_base + lane]
                    acc_lo += x[out_base + lane] * (d1 * np.float32(qbyte & 0x0F) - mm1)
                    acc_hi += x[out_base + 32 + lane] * (d2 * np.float32(qbyte >> 4) - mm2)
        y[o] = acc_lo + acc_hi
    return y


def qgemv_q4_k(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    """`raw` holds `out_features * (in_features // 256)` real Q4_K superblocks, row-major (every
    block of one output row before the next), exactly as `Q4_KStrategy.dequantize` already expects
    - see `GGUFModelLoader.load_tensor`'s own reshape for why that's the real on-disk order.

    Header parsing (d/dmin f16->f32, scales/qs byte slicing) stays in numpy - cheap, real profiling
    this session confirmed the expensive part is the per-element nibble unpack + accumulate below,
    not this. Returns a real `(out_features,)` float32 tensor: `y = x @ W.T`.
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
                # Four independent accumulator chains (acc1-acc4, one per real q1-q4 value) -
                # see qgemv_q4_k's own kernel comment for why a register-only design (no
                # heap-allocated unpack/accumulate scratch buffer) is what real re-benchmarking
                # found actually helped here.
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
    """Same shape/layout contract as `qgemv_q4_k`, for Q6_K blocks (see `Q6_KStrategy` in
    kquants.py for the real per-block field layout this mirrors)."""
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
