import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.fused_attention import fused_cached_attention
from app.architectures.layers import RMSNorm, SwiGLUMLP
from app.architectures.rope import apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache


def _causal_mask(
    q_len: int, kv_len: int, cache_offset: int, device: torch.device | None = None
) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class QwenAttention(nn.Module):
    """`GroupedQueryAttention` (`mistral3_layers.py`) plus the deltas Qwen2/Qwen3 need, shared here
    like `GraniteAttention`:

    `qkv_bias`: Qwen2 biases `q_proj`/`k_proj`/`v_proj` (HF `modeling_qwen2.py`) but never `o_proj`;
    Qwen3 has none (`config.attention_bias = False`).

    `qk_norm_eps`: Qwen3's per-head-dim `RMSNorm` on q and k after the reshape to heads, before RoPE
    (`modeling_qwen3.py`). RMSNorm's last-dim reduction makes before/after the transpose identical,
    so it is applied after. `None` for Qwen2.

    `o_proj_bias`: Qwen2/Qwen3/Command-R never bias `o_proj` (default `False`); StarCoder2 is the
    first reuse that does (`config.use_bias` gates all four projections; `Starcoder2Attention`).
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
        o_proj_bias: bool = False,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=qkv_bias, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=qkv_bias, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=qkv_bias, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=o_proj_bias, dtype=dtype)
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

        # enable_gqa: SDPA shares each k/v head across its query-head group itself, with no
        # repeat_interleave copy of the cached k/v per step
        # (measured on the dev machine: at 200 tokens that copy cost ~90 ms per generated token
        # over 28 layers vs ~7 ms without; the gap grows with the conversation).
        if kv_cache is not None:
            out = fused_cached_attention(q, k, v, cache_offset)
            if out is None:
                mask = _causal_mask(seq_len, k.shape[-2], cache_offset, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class QwenDecoderLayer(nn.Module):
    """Same pre-norm two-block layout as `Mistral3DecoderLayer`, with no Granite-style residual
    scaling and no MoE injection (MLP is always
    a plain `SwiGLUMLP` for Qwen2 and Qwen3), so it builds its attention/MLP internally instead of
    taking them as arguments.
    """

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
