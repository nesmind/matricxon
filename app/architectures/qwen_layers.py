import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm, SwiGLUMLP
from app.architectures.rope import apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache


def _causal_mask(q_len: int, kv_len: int, cache_offset: int) -> torch.Tensor:
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class QwenAttention(nn.Module):
    """`GroupedQueryAttention` (`mistral3_layers.py`) plus the two real deltas Qwen2/Qwen3 need,
    shared here rather than duplicated per architecture (same "one flexible class" precedent
    `GraniteAttention` already established for a comparably small real delta):

    `qkv_bias` - real Qwen2 checkpoints have a real bias on `q_proj`/`k_proj`/`v_proj` (hardcoded
    `bias=True` in HF's own `modeling_qwen2.py`, not config-gated) but never on `o_proj`; real
    Qwen3 checkpoints have no bias anywhere (`config.attention_bias = False` on every real small
    checkpoint, confirmed no `attn_*.bias` tensors exist in a real downloaded GGUF file at all).

    `qk_norm_eps` - Qwen3's real structural delta from Qwen2: a per-head-dim `RMSNorm` applied to
    both q and k, right after the reshape-to-heads split, **before** RoPE (confirmed exact
    placement from `modeling_qwen3.py`: `q_norm(q_proj(x).view(...))` then `transpose` then
    `apply_rotary_pos_emb` - RMSNorm's last-dim reduction makes applying it before or after the
    transpose used here mathematically identical, so this applies it after, matching every other
    call in this class). `None` for Qwen2 (no QK-norm at all).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        qkv_bias: bool,
        qk_norm_eps: float | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=qkv_bias, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=qkv_bias, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=qkv_bias, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)
        if qk_norm_eps is not None:
            self.q_norm = RMSNorm(head_dim, qk_norm_eps, dtype=dtype)
            self.k_norm = RMSNorm(head_dim, qk_norm_eps, dtype=dtype)
        else:
            self.q_norm = None
            self.k_norm = None

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

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
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


class QwenDecoderLayer(nn.Module):
    """Same pre-norm two-block layout as `Mistral3DecoderLayer` - no Granite-style residual
    scaling, no MoE injection (MLP is always plain `SwiGLUMLP` for both Qwen2 and Qwen3), so this
    builds its own attention/MLP internally rather than taking them as constructor arguments."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        ffn_len: int,
        rms_eps: float,
        qkv_bias: bool,
        qk_norm_eps: float | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = QwenAttention(
            n_embd, n_head, n_head_kv, head_dim, qkv_bias, qk_norm_eps, dtype=dtype
        )
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
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, layer_idx)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x
