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


def gated_delta_rule_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same maths as `gated_delta_rule_reference` (same shapes, `q` already scaled), but the T
    tokens are processed in chunks of `chunk`: inside a chunk the recurrence becomes a few batched
    matmuls (a small triangular solve, built row by row), and only the state hand-off between
    chunks is sequential - so a long prefill on a GPU is T/chunk steps instead of T. Algorithm of
    HF's `torch_chunk_gated_delta_rule`; pure torch, so it runs on any device."""
    n_tokens = q.shape[0]
    pad = (-n_tokens) % chunk

    # (T, H, D) -> (H, n_chunks, chunk, D), zero-padded: a padded token has beta 0 (no write) and
    # g 0 (no decay), so it leaves the state alone and its output rows are dropped below.
    def heads(x: torch.Tensor) -> torch.Tensor:
        x = (
            torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad))
            if x.dim() == 3
            else (torch.nn.functional.pad(x, (0, 0, 0, pad)))
        )
        return x.transpose(0, 1).reshape(x.shape[1], -1, chunk, *x.shape[2:])

    q, k, v = heads(q), heads(k), heads(v)
    g, beta = heads(g), heads(beta)  # (H, nc, chunk)
    g = g.cumsum(dim=-1)
    k_beta, v_beta = k * beta.unsqueeze(-1), v * beta.unsqueeze(-1)
    eye = torch.eye(chunk, device=q.device)
    upper = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
    decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * decay).masked_fill(upper, 0)
    for i in range(1, chunk):  # forward substitution for (I + A)^-1, one row at a time
        row = attn[..., i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + eye
    v_corrected = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    state = state.clone()
    strict_upper = torch.triu(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device), 1)
    outputs = torch.empty_like(v_corrected)
    for c in range(q.shape[1]):
        local = ((q[:, c] @ k[:, c].transpose(-1, -2)) * decay[:, c]).masked_fill(strict_upper, 0)
        v_new = v_corrected[:, c] - k_cumdecay[:, c] @ state
        outputs[:, c] = (q[:, c] * g[:, c].unsqueeze(-1).exp()) @ state + local @ v_new
        last = g[:, c, -1]
        state = (
            state * last.exp().view(-1, 1, 1)
            + (k[:, c] * (last.unsqueeze(-1) - g[:, c]).exp().unsqueeze(-1)).transpose(-1, -2)
            @ v_new
        )
    out = outputs.reshape(outputs.shape[0], -1, outputs.shape[-1])[:, :n_tokens]
    return out.transpose(0, 1).contiguous(), state


def gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Same contract as `gated_delta_rule_reference`; `state` is updated in place and returned
    (on the CPU). Tensors on a GPU run in torch instead - the compiled kernels read CPU memory -
    and return a new state: the chunked form for a prefill, the plain step for one token."""
    if q.device.type != "cpu":
        fn = gated_delta_rule_chunked if q.shape[0] > 1 else gated_delta_rule_reference
        return fn(q.float(), k.float(), v.float(), g.float(), beta.float(), state.float())
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
