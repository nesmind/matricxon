import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.rope import apply_rotary_pos_emb_partial
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(q_len: int, kv_len: int, cache_offset: int) -> torch.Tensor:
    """Same causal rule as `Mistral3DecoderLayer`'s own `_causal_mask` (duplicated, not imported -
    same precedent `Gemma4DecoderLayer`'s own `_causal_mask` already set: this project's per-
    architecture layers modules keep their own copy of small helpers like this rather than
    cross-importing another module's private name): query i (local index) may attend to key j iff
    j <= i + cache_offset."""
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class Phi2Attention(nn.Module):
    """Plain (non-grouped) multi-head attention with real biases on every projection - a real,
    confirmed-live Phi-2 GGUF (moondream2's text half) has `attention.head_count ==
    attention.head_count_kv` always (no GQA) and a real bias tensor on `attn_qkv`/`attn_output`,
    unlike every bias-free decoder matricxon has today.

    `q_proj`/`k_proj`/`v_proj` are kept as three separate `nn.Linear`s even though the real GGUF
    tensor they're loaded from (`attn_qkv`) is one fused matrix - see
    `Phi2Architecture._materialize_weights`'s own comment for why, and for the one real,
    unconfirmed-from-metadata-alone open question this shape is deliberately left able to absorb
    (`unpermute_rope_rows` on just the q/k sub-blocks, if real-weight verification needs it).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int,
        rope_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head * head_dim, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head * head_dim, bias=True, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=True, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb_partial(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
            # See GroupedQueryAttention.forward's identical comment - a mixed per-layer dtype
            # plan can leave this layer's own q/k/v dtype differing from the cache's fixed one.
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        else:
            cache_offset = 0

        if kv_cache is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class Phi2MLP(nn.Module):
    """`down_proj(gelu(up_proj(x)))` - a plain, non-gated GELU MLP with real biases on both
    projections, confirmed via a real Phi-2 GGUF's `ffn_up`/`ffn_down` bias tensors. Its own class
    rather than reusing `SwiGLUMLP` (different formula entirely: no gate, and always biased where
    `SwiGLUMLP` defaults bias-free)."""

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Phi2DecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))` - one shared `LayerNorm` call
    feeds *both* branches from the same input, added back together. Real Phi-2 structure
    (confirmed via a real moondream2 GGUF: only one `attn_norm` tensor per layer, no separate
    post-attention norm) - structurally different from every other decoder layer here, which is
    sequential (attention first, its own norm, then the MLP - see `Mistral3DecoderLayer`)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int,
        rope_dim: int,
        ffn_len: int,
        layer_norm_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)
        self.self_attn = Phi2Attention(n_embd, n_head, head_dim, rope_dim, dtype=dtype)
        self.mlp = Phi2MLP(n_embd, ffn_len, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)

        stage_started = time.monotonic()
        attn_out = self.self_attn(h, cos, sin, kv_cache, layer_idx)
        logger.debug(
            "  layer %s: attention %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )

        stage_started = time.monotonic()
        mlp_out = self.mlp(h)
        logger.debug("  layer %s: ffn %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000)

        return x + attn_out + mlp_out
