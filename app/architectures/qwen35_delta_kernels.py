"""Fused Gated DeltaNet recurrence for Qwen3.5 - one Numba kernel and one C kernel
(`app/native/src/mx_delta_rule.c`) computing the identical float32 math, kept in sync (see
CLAUDE.md). `gated_delta_rule` picks the native one when `Settings.gemv_backend == "native"` built
fine, else Numba; `gated_delta_rule_reference` is the plain torch loop both are tested against.

Per token and head (state S: d_k x d_v): S *= exp(g); delta = (v - S^T k) * beta; S += k (x) delta;
out = S^T q. Heads are independent across time, so both kernels parallelize over heads.
"""

import numba
import numpy as np
import torch

from app.native.gemm import NativeGemm


@numba.njit(parallel=True, cache=True)
def _numba_delta_rule(q, k, v, g, beta, state, out):  # pragma: no cover - jitted
    n_tokens, n_heads, dk = q.shape
    dv = v.shape[2]
    for h in numba.prange(n_heads):
        s = state[h]
        delta = np.empty(dv, dtype=np.float32)
        for t in range(n_tokens):
            decay = np.float32(np.exp(g[t, h]))
            delta[:] = 0.0
            for j in range(dk):
                kj = k[t, h, j]
                for i in range(dv):
                    s[j, i] *= decay
                    delta[i] += s[j, i] * kj
            for i in range(dv):
                delta[i] = (v[t, h, i] - delta[i]) * beta[t, h]
            out[t, h, :] = 0.0
            for j in range(dk):
                kj = k[t, h, j]
                qj = q[t, h, j]
                for i in range(dv):
                    s[j, i] += kj * delta[i]
                    out[t, h, i] += s[j, i] * qj


def gated_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain per-token torch loop. q/k (T,H,dk), v (T,H,dv), g/beta (T,H), state (H,dk,dv),
    all float32. Returns (outputs (T,H,dv), new state)."""
    outputs = torch.empty_like(v)
    for t in range(q.shape[0]):
        state = state * torch.exp(g[t]).view(-1, 1, 1)
        predicted = torch.einsum("hkv,hk->hv", state, k[t])
        delta = (v[t] - predicted) * beta[t].view(-1, 1)
        state = state + torch.einsum("hk,hv->hkv", k[t], delta)
        outputs[t] = torch.einsum("hkv,hk->hv", state, q[t])
    return outputs, state


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as `gated_delta_rule_reference`; `state` is updated in place and returned."""
    q, k, v, g, beta = (t.detach().float().contiguous() for t in (q, k, v, g, beta))
    state = state.detach().float().contiguous()
    out = torch.empty_like(v)
    native = NativeGemm.active()
    if native is not None:
        native.gated_delta_rule(q, k, v, g, beta, state, out)
    else:
        _numba_delta_rule(
            q.numpy(), k.numpy(), v.numpy(), g.numpy(), beta.numpy(), state.numpy(), out.numpy()
        )
    return out, state
