import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.fused_attention import fused_cached_attention
from app.architectures.rope import apply_rotary_pos_emb_partial
from app.architectures.shared_projection import project_shared
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(
    q_len: int, kv_len: int, cache_offset: int, device: torch.device | None = None
) -> torch.Tensor:
    """Same causal rule as `Mistral3DecoderLayer`'s `_causal_mask` (duplicated, not imported, as
    `Gemma4DecoderLayer`'s: per-architecture
    layers modules keep their own copy of small helpers instead of importing another module's
    private name): query i (local index) attends to key j iff j <= i + cache_offset.
    """
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class Phi2Attention(nn.Module):
    """Plain multi-head attention with biases on every projection: a Phi-2 GGUF always has
    `head_count == head_count_kv` (no GQA) and a bias on `attn_qkv`/`attn_output`, unlike
    matricxon's other bias-free decoders.

    `q_proj`/`k_proj`/`v_proj` stay three `nn.Linear`s though loaded from one fused `attn_qkv`; see
    `Phi2Architecture._materialize_weights` for why, and for the open question this shape absorbs
    (`unpermute_rope_rows` on just the q/k sub-blocks, if real weights need it).
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

        q, k, v = project_shared(self, ("q_proj", "k_proj", "v_proj"), x)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb_partial(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
            # See GroupedQueryAttention.forward: a mixed per-layer dtype plan can leave this layer's
            # q/k/v dtype different from the cache's fixed one.
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        else:
            cache_offset = 0

        if kv_cache is not None:
            out = fused_cached_attention(q, k, v, cache_offset)
            if out is None:
                mask = _causal_mask(seq_len, k.shape[-2], cache_offset, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class Phi2MLP(nn.Module):
    """`down_proj(gelu(up_proj(x)))`: a plain non-gated GELU MLP with biases on both projections (a
    Phi-2 GGUF's `ffn_up`/`ffn_down`
    bias tensors). Its own class, not `SwiGLUMLP` (different formula: no gate, always biased where
    `SwiGLUMLP` defaults bias-free).
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Phi2DecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))`: one shared `LayerNorm` call feeds
    *both* branches from the same
    input. Phi-2's structure (a moondream2 GGUF has one `attn_norm` per layer, no post-attention
    norm), unlike every other decoder
    layer here, which is sequential (attention, its norm, then MLP; `Mistral3DecoderLayer`).
    """

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
