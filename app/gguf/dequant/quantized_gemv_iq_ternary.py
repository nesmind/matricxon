"""GEMV kernels for IQ4_NL, IQ4_XS, TQ1_0, TQ2_0 - same real, honest performance profile as
`quantized_gemv_extended.py` (register-only accumulators, no heap scratch buffer). Correctness
cross-checked against each type's own already-oracle-validated `QuantStrategy.dequantize()` in
`iq_ternary.py`.

`_KVALUES_IQ4NL`/`_POW3` are passed into the njit kernels as explicit arguments rather than
captured as module globals - keeps the kernel's inputs fully explicit, same reasoning
`_get_scale_min_k4`-style small helpers already follow in this package.
"""

import numba
import numpy as np
import torch

_KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32,
)
_POW3 = np.array([1, 3, 9, 27, 81, 243], dtype=np.uint8)

IQ4_NL_BLOCK = 32
IQ4_NL_TYPE_SIZE = 2 + 16  # 18
_QK_K = 256
IQ4_XS_TYPE_SIZE = 2 + 2 + 4 + 128  # 136
TQ1_0_TYPE_SIZE = 48 + 4 + 2  # 54
TQ2_0_TYPE_SIZE = 64 + 2  # 66


def _require_divisible(in_features: int, block_size: int) -> int:
    if in_features % block_size != 0:
        raise ValueError(f"in_features={in_features} is not a multiple of block_size={block_size}")
    return in_features // block_size


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq4_nl_kernel(x, d, qs, out_features, n_blocks, kvalues):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for b in range(n_blocks):
            x_base = b * IQ4_NL_BLOCK
            d_val = d[o, b]
            for j in range(16):
                qbyte = qs[o, b, j]
                lo = kvalues[qbyte & 0x0F]
                hi = kvalues[qbyte >> 4]
                acc += x[x_base + j] * (d_val * lo) + x[x_base + 16 + j] * (d_val * hi)
        y[o] = acc
    return y


def qgemv_iq4_nl(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_blocks = _require_divisible(in_features, IQ4_NL_BLOCK)
    count = out_features * n_blocks * IQ4_NL_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_blocks, IQ4_NL_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_blocks)
    qs = np.ascontiguousarray(blocks[:, :, 2 : 2 + 16])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    kernel = _qgemv_iq4_nl_kernel(x_np, d, qs, out_features, n_blocks, _KVALUES_IQ4NL)
    return torch.from_numpy(kernel)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_iq4_xs_kernel(x, d, scales_h, scales_l, qs, out_features, n_super, kvalues):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            sh = scales_h[o, sb]
            for ib in range(8):
                sl_byte = scales_l[o, sb, ib // 2]
                ls_lo = np.int32((sl_byte >> (4 * (ib % 2))) & 0x0F)
                ls_hi = np.int32((sh >> (2 * ib)) & 3) << 4
                dl = d_val * np.float32((ls_lo | ls_hi) - 32)
                q_base = ib * 16
                out_base = x_base + ib * 32
                for j in range(16):
                    qbyte = qs[o, sb, q_base + j]
                    lo = kvalues[qbyte & 0x0F]
                    hi = kvalues[qbyte >> 4]
                    acc += x[out_base + j] * (dl * lo) + x[out_base + 16 + j] * (dl * hi)
        y[o] = acc
    return y


def qgemv_iq4_xs(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * IQ4_XS_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, IQ4_XS_TYPE_SIZE
    )
    d = blocks[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    scales_h = blocks[:, :, 2:4].copy().view("<u2").reshape(out_features, n_super)
    scales_l = np.ascontiguousarray(blocks[:, :, 4:8])
    qs = np.ascontiguousarray(blocks[:, :, 8 : 8 + 128])
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(
        _qgemv_iq4_xs_kernel(x_np, d, scales_h, scales_l, qs, out_features, n_super, _KVALUES_IQ4NL)
    )


@numba.njit(inline="always")
def _tq_digit(byte, power):
    """byte, power: uint8 scalars. Mirrors ggml's `((uint16_t)(byte*p) * 3) >> 8`
    base-3-digit-extraction trick - the uint8 multiply wraps mod 256 exactly like a real C
    `uint8_t`. Returns the digit in {0, 1, 2} as float32."""
    q = np.uint8(byte * power)
    return np.float32((np.uint16(q) * np.uint16(3)) >> 8)


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_tq1_0_kernel(x, d, qs, qh, out_features, n_super, pow3):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            col = 0
            for n in range(5):
                p = pow3[n]
                for m in range(32):
                    val = (_tq_digit(qs[o, sb, m], p) - 1.0) * d_val
                    acc += x[x_base + col] * val
                    col += 1
            for n in range(5):
                p = pow3[n]
                for m in range(16):
                    val = (_tq_digit(qs[o, sb, 32 + m], p) - 1.0) * d_val
                    acc += x[x_base + col] * val
                    col += 1
            for n in range(4):
                p = pow3[n]
                for m in range(4):
                    val = (_tq_digit(qh[o, sb, m], p) - 1.0) * d_val
                    acc += x[x_base + col] * val
                    col += 1
        y[o] = acc
    return y


def qgemv_tq1_0(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * TQ1_0_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, TQ1_0_TYPE_SIZE
    )
    qs = np.ascontiguousarray(blocks[:, :, 0:48])
    qh = np.ascontiguousarray(blocks[:, :, 48:52])
    d = blocks[:, :, 52:54].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_tq1_0_kernel(x_np, d, qs, qh, out_features, n_super, _POW3))


@numba.njit(parallel=True, cache=True, fastmath=True)
def _qgemv_tq2_0_kernel(x, d, qs, out_features, n_super):
    y = np.zeros(out_features, dtype=np.float32)
    for o in numba.prange(out_features):
        acc = np.float32(0.0)
        for sb in range(n_super):
            x_base = sb * _QK_K
            d_val = d[o, sb]
            col = 0
            for j in range(0, 64, 32):
                for shift_idx in range(4):
                    shift = shift_idx * 2
                    for m in range(32):
                        bits = np.float32((qs[o, sb, j + m] >> shift) & 3)
                        acc += x[x_base + col] * ((bits - 1.0) * d_val)
                        col += 1
        y[o] = acc
    return y


def qgemv_tq2_0(
    x: torch.Tensor, raw: memoryview, out_features: int, in_features: int
) -> torch.Tensor:
    n_super = _require_divisible(in_features, _QK_K)
    count = out_features * n_super * TQ2_0_TYPE_SIZE
    blocks = np.frombuffer(raw, dtype=np.uint8, count=count).reshape(
        out_features, n_super, TQ2_0_TYPE_SIZE
    )
    qs = np.ascontiguousarray(blocks[:, :, 0:64])
    d = blocks[:, :, 64:66].copy().view("<f2").astype(np.float32).reshape(out_features, n_super)
    x_np = np.ascontiguousarray(x.detach().numpy(), dtype=np.float32)
    return torch.from_numpy(_qgemv_tq2_0_kernel(x_np, d, qs, out_features, n_super))
