"""Numba twins of the C kernels in app/native/src/mx_fused_ops.c, mx_attention.c and mx_sample.c -
the same maths, kept in sync with them (see CLAUDE.md). Used when `Settings.gemv_backend` is
"numba" or the C library didn't build. Layouts match the C versions: see their header comments."""

import math

import numba
import numpy as np


@numba.njit(cache=True, fastmath=True)
def rms_norm(x, w, out, eps):  # pragma: no cover - jitted
    rows, dim = x.shape
    for r in range(rows):
        total = np.float32(0.0)
        for i in range(dim):
            total += x[r, i] * x[r, i]
        scale = np.float32(1.0) / np.float32(math.sqrt(total / dim + eps))
        for i in range(dim):
            out[r, i] = x[r, i] * scale * w[i]


@numba.njit(cache=True)
def rope(x, out, cos, sin):  # pragma: no cover - jitted
    n_heads, n_tokens, dim = x.shape
    half = dim // 2
    for h in range(n_heads):
        for t in range(n_tokens):
            for i in range(half):
                out[h, t, i] = x[h, t, i] * cos[t, i] - x[h, t, i + half] * sin[t, i]
                out[h, t, i + half] = (
                    x[h, t, i + half] * cos[t, i + half] + x[h, t, i] * sin[t, i + half]
                )


@numba.njit(cache=True, fastmath=True)
def attention(q, k, v, out, scale, offset, window):  # pragma: no cover - jitted
    """Serial on purpose (a `prange` over heads measured slower at decode sizes: the thread
    hand-off dominates), and `fastmath` so the dot-product reductions vectorize."""
    n_heads, n_q, dim = q.shape
    group = n_heads // k.shape[0]
    n_kv = k.shape[1]
    scores = np.empty(n_kv, dtype=np.float32)
    for h in range(n_heads):
        kv = h // group
        for t in range(n_q):
            limit = min(t + offset + 1, n_kv)
            start = max(t + offset - window + 1, 0) if window > 0 else 0
            peak = np.float32(-np.inf)
            for j in range(start, limit):
                dot = np.float32(0.0)
                for i in range(dim):
                    dot += q[h, t, i] * k[kv, j, i]
                scores[j] = dot * scale
                peak = max(peak, scores[j])
            total = np.float32(0.0)
            for j in range(start, limit):
                scores[j] = np.float32(math.exp(scores[j] - peak))
                total += scores[j]
            for i in range(dim):
                out[h, t, i] = 0.0
            for j in range(start, limit):
                p = scores[j] / total
                for i in range(dim):
                    out[h, t, i] += p * v[kv, j, i]


@numba.njit(cache=True)
def _kth_largest(values, k):  # pragma: no cover - jitted
    """Smallest of the k largest values: a size-k min-heap over one pass."""
    heap = np.empty(k, dtype=np.float32)
    size = 0
    for idx in range(values.shape[0]):
        x = values[idx]
        if size < k:
            c = size
            size += 1
            while c > 0 and heap[(c - 1) // 2] > x:
                heap[c] = heap[(c - 1) // 2]
                c = (c - 1) // 2
            heap[c] = x
        elif x > heap[0]:
            c = 0
            while True:
                left = 2 * c + 1
                right = left + 1
                m = c
                lowest = x
                if left < size and heap[left] < lowest:
                    m = left
                    lowest = heap[left]
                if right < size and heap[right] < lowest:
                    m = right
                if m == c:
                    break
                heap[c] = heap[m]
                c = m
            heap[c] = x
    return heap[0]


@numba.njit(cache=True)
def sample(logits, penalty_ids, penalty, temperature, top_k, top_p, u):  # pragma: no cover
    n_vocab = logits.shape[0]
    work = logits.copy()
    for i in range(penalty_ids.shape[0]):
        x = work[penalty_ids[i]]
        work[penalty_ids[i]] = x / penalty if x > 0 else x * penalty
    work /= temperature
    floor_value = np.float32(-np.inf)
    if top_k > 0 and top_k < n_vocab:
        floor_value = _kth_largest(work, top_k)
    ids = np.nonzero(work >= floor_value)[0]
    values = work[ids]
    peak = values.max()
    weights = np.exp(values - peak)
    total = weights.sum()
    if top_p < 1.0:
        order = np.argsort(-weights)
        ids, weights = ids[order], weights[order]
        before = 0.0
        keep = 0
        while keep < weights.shape[0] and before / total <= top_p:
            before += weights[keep]
            keep += 1
        ids, weights, total = ids[:keep], weights[:keep], before
    target = u * total
    running = 0.0
    for c in range(weights.shape[0]):
        running += weights[c]
        if running > target:
            return ids[c]
    return ids[-1]
