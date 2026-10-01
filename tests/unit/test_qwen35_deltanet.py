"""`Qwen35GatedDeltaNet` / `gated_delta_rule` against an independent per-token reference
(explicit loops, no einsum, no F.conv1d) - same treatment as test_nemotron_h_mamba2_mixer.py -
plus the prefill/decode state-carry invariant the KV-less recurrent cache depends on.
"""

import pytest
import torch
import torch.nn.functional as F

from app.architectures.qwen35_deltanet import Qwen35GatedDeltaNet
from app.runtime.mamba_cache import NemotronHHybridCache

N_EMBD, N_K, N_V, HD, KERNEL, EPS = 6, 2, 4, 3, 4, 1e-6
CONV_DIM = (2 * N_K + N_V) * HD


def _mixer(tiled: bool = True, seed: int = 0) -> Qwen35GatedDeltaNet:
    torch.manual_seed(seed)
    m = Qwen35GatedDeltaNet(N_EMBD, N_K, N_V, HD, KERNEL, EPS, tiled_heads=tiled)
    with torch.no_grad():
        for p in (m.qkv_proj, m.z_proj, m.beta_proj, m.alpha_proj, m.out_proj):
            p.weight.copy_(torch.randn_like(p.weight) * 0.3)
        m.conv1d_weight.copy_(torch.randn_like(m.conv1d_weight) * 0.3)
        m.dt_bias.copy_(torch.randn_like(m.dt_bias) * 0.1)
        m.a.copy_(-torch.rand_like(m.a) * 2 - 0.01)
        m.norm_weight.copy_(1 + torch.randn_like(m.norm_weight) * 0.05)
    return m


def _cache() -> NemotronHHybridCache:
    cache = NemotronHHybridCache(
        layer_types=["mamba"],
        attention_layer_shape=(1, 1),
        mamba_conv_state_shape=(KERNEL - 1, CONV_DIM),
        mamba_ssm_state_shape=(N_V, HD, HD),
        max_seq_len=32,
    )
    return cache


def _l2(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _reference(x: torch.Tensor, m: Qwen35GatedDeltaNet, tiled: bool) -> torch.Tensor:
    seq = x.shape[1]
    qkv = x[0] @ m.qkv_proj.weight.T
    history = torch.zeros(KERNEL - 1, CONV_DIM)
    conv = torch.empty(seq, CONV_DIM)
    for t in range(seq):
        window = torch.cat([history, qkv[t : t + 1]], dim=0)
        conv[t] = F.silu((window.T * m.conv1d_weight).sum(-1))
        history = window[1:]
    q = conv[:, : N_K * HD].view(seq, N_K, HD)
    k = conv[:, N_K * HD : 2 * N_K * HD].view(seq, N_K, HD)
    v = conv[:, 2 * N_K * HD :].view(seq, N_V, HD)
    z = (x[0] @ m.z_proj.weight.T).view(seq, N_V, HD)
    beta = torch.sigmoid(x[0] @ m.beta_proj.weight.T)
    g = m.a * F.softplus(x[0] @ m.alpha_proj.weight.T + m.dt_bias)
    state = torch.zeros(N_V, HD, HD)
    out = torch.empty(seq, N_V, HD)
    for t in range(seq):
        for h in range(N_V):
            kh = h % N_K if tiled else h // (N_V // N_K)
            qt = _l2(q[t, kh]) * HD**-0.5
            kt = _l2(k[t, kh])
            s = state[h] * torch.exp(g[t, h])
            delta = (v[t, h] - s.T @ kt) * beta[t, h]
            s = s + torch.outer(kt, delta)
            state[h] = s
            o = s.T @ qt
            o = o * torch.rsqrt(o.pow(2).mean() + EPS) * m.norm_weight
            out[t, h] = o * F.silu(z[t, h])
    return (out.reshape(seq, N_V * HD) @ m.out_proj.weight.T).unsqueeze(0)


@pytest.mark.parametrize("tiled", [True, False])
def test_matches_reference_recurrence(tiled: bool) -> None:
    m = _mixer(tiled)
    x = torch.randn(1, 7, N_EMBD) * 0.5
    with torch.no_grad():
        got = m(x, _cache(), 0)
    assert torch.allclose(got, _reference(x, m, tiled), atol=1e-5)


def test_head_expansion_order_matters() -> None:
    x = torch.randn(1, 5, N_EMBD)
    with torch.no_grad():
        tiled = _mixer(True)(x, _cache(), 0)
        interleaved = _mixer(False)(x, _cache(), 0)
    assert not torch.allclose(tiled, interleaved, atol=1e-4)


def test_prefill_then_decode_equals_one_shot() -> None:
    m = _mixer()
    x = torch.randn(1, 9, N_EMBD) * 0.5
    with torch.no_grad():
        full = m(x, _cache(), 0)
        cache = _cache()
        parts = [m(x[:, :5], cache, 0)] + [m(x[:, t : t + 1], cache, 0) for t in range(5, 9)]
    assert torch.allclose(torch.cat(parts, dim=1), full, atol=1e-5)
