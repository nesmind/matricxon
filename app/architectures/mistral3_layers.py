import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm, SwiGLUMLP
from app.architectures.rope import apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(q_len: int, kv_len: int, cache_offset: int) -> torch.Tensor:
    """query i (local index) may attend to key j iff j <= i + cache_offset -
    i.e. its own absolute position and everything before it. `cache_offset=0`
    and `q_len == kv_len` reduces to plain causal masking (prefill); a
    one-token decode step (`q_len=1`) attends to the whole existing cache
    plus itself.
    """
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)

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
        k = self.k_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
            # KVCache stores every layer in one fixed dtype decided by its own caller (see
            # ChatEngine.stream) - usually not an issue when every layer shares one model-wide
            # dtype, but under mixed per-layer dtypes (see ModelManager._load's own docstring)
            # that fixed cache dtype can differ from *this* layer's own q/k/v dtype.
            # scaled_dot_product_attention requires q/k/v to share one dtype (unlike RMSNorm's
            # elementwise ops, it does not implicitly promote a mismatch) - cast to q's dtype here,
            # the same defensive pattern apply_rotary_pos_emb already uses for cos/sin.
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        else:
            cache_offset = 0

        n_rep = self.n_head // self.n_head_kv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        if kv_cache is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class Mistral3DecoderLayer(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        ffn_len: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = GroupedQueryAttention(n_embd, n_head, n_head_kv, head_dim, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mlp = SwiGLUMLP(n_embd, ffn_len, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        stage_started = time.monotonic()
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, layer_idx)
        logger.debug(
            "  layer %s: attention %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )

        stage_started = time.monotonic()
        x = x + self.mlp(self.post_attention_layernorm(x))
        logger.debug(
            "  layer %s: ffn %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )
        return x
