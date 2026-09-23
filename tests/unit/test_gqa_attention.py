"""GroupedQueryAttention leaves the k/v-head sharing to SDPA (`enable_gqa=True`) instead of
copying the cached k/v with repeat_interleave every step. Checked here against that explicit
repeat_interleave computation - full-sequence, and prefill + one-token decode through a KVCache.
"""

import torch
import torch.nn.functional as F

from app.architectures.mistral3_layers import GroupedQueryAttention
from app.architectures.rope import RotaryEmbedding, apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache

N_EMBD, N_HEAD, N_HEAD_KV, HEAD_DIM = 32, 4, 2, 8


def _attention() -> GroupedQueryAttention:
    torch.manual_seed(0)
    attn = GroupedQueryAttention(N_EMBD, N_HEAD, N_HEAD_KV, HEAD_DIM)
    for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        torch.nn.init.normal_(proj.weight, std=0.2)
    return attn


def _reference(
    attn: GroupedQueryAttention, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Plain causal attention over the whole sequence with every k/v head repeated explicitly."""
    seq_len = x.shape[1]
    q = attn.q_proj(x).view(1, seq_len, N_HEAD, HEAD_DIM).transpose(1, 2)
    k = attn.k_proj(x).view(1, seq_len, N_HEAD_KV, HEAD_DIM).transpose(1, 2)
    v = attn.v_proj(x).view(1, seq_len, N_HEAD_KV, HEAD_DIM).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    rep = N_HEAD // N_HEAD_KV
    out = F.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1), is_causal=True
    )
    return attn.o_proj(out.transpose(1, 2).reshape(1, seq_len, N_HEAD * HEAD_DIM))


def test_gqa_matches_explicit_repeat_with_and_without_cache() -> None:
    attn = _attention()
    rope = RotaryEmbedding(head_dim=HEAD_DIM, rope_theta=10000.0)
    x = torch.randn(1, 6, N_EMBD)
    cos, sin = rope(torch.arange(6))

    with torch.no_grad():
        expected = _reference(attn, x, cos, sin)
        full = attn(x, cos, sin)

        cache = KVCache([(N_HEAD_KV, HEAD_DIM)], max_seq_len=16)
        prefill = attn(x[:, :5], cos[:5], sin[:5], cache, 0)
        cache.advance(5)
        decode = attn(x[:, 5:], cos[5:], sin[5:], cache, 0)

    assert torch.allclose(full, expected, atol=1e-5)
    assert torch.allclose(prefill, expected[:, :5], atol=1e-5)
    assert torch.allclose(decode, expected[:, 5:], atol=1e-5)
