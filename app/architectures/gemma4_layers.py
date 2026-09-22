import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm
from app.architectures.rope import apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(q_len: int, kv_len: int, cache_offset: int, window: int | None) -> torch.Tensor:
    """Same causal rule as Mistral3DecoderLayer's own `_causal_mask`, plus an

    optional sliding window: query i may attend to key j iff
    `j <= i + cache_offset` (causal) and, if `window` is set,
    `j > i + cache_offset - window` (Gemma4's local/sliding-window layers -
    `window` comes straight from the real `gemma4.attention.sliding_window`
    metadata, confirmed 1024 on the real model this was built against).
    """
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    causal = kv_idx <= (q_idx + cache_offset)
    if window is None:
        return causal
    return causal & (kv_idx > (q_idx + cache_offset - window))


def _rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma4's `v_norm` - a real, active step (not an approximation

    skipped for simplicity) but with no learnable weight at all
    (`with_scale=False` in the real HF `Gemma4RMSNorm`, confirmed by this
    GGUF carrying no `attn_v_norm.weight` tensor anywhere), so it's a
    plain function here rather than a module with nothing to load.
    """
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(input_dtype)


class Gemma4MLP(nn.Module):
    """Gated FFN with GELU-tanh activation (real HF default

    `hidden_activation="gelu_pytorch_tanh"`, confirmed - NOT SiLU, so this
    can't reuse the shared `SwiGLUMLP`, despite the otherwise-identical
    gate/up/down shape).
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.gate_proj(x), approximate="tanh")
        return self.down_proj(gate * self.up_proj(x))


class Gemma4Attention(nn.Module):
    """QK-norm GQA attention with per-layer-type head_dim/kv-heads (M10's

    real finding: local/sliding layers use 8 kv-heads of dim 256, global
    layers use 1 kv-head - effectively MQA - of dim 512, confirmed via this
    GGUF's actual per-layer tensor shapes, not assumed uniform). Global
    layers additionally reuse the *raw* (pre-`k_norm`, pre-RoPE) key
    projection as their value input rather than having their own
    `v_proj` at all (confirmed: no `attn_v.weight` tensor exists for a
    real global-layer index in this GGUF) - `use_v_from_k` below.

    `scaling=1.0` (not the usual `1/sqrt(head_dim)`) is a real, confirmed
    Gemma4 choice, not an oversight - QK-norm already keeps query/key
    magnitudes controlled, apparently making the traditional scaling
    redundant for this architecture.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        rms_eps: float,
        use_v_from_k: bool,
        sliding_window: int | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rms_eps = rms_eps
        self.use_v_from_k = use_v_from_k
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.v_proj = (
            None
            if use_v_from_k
            else nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        )
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_norm(self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim))
        q = q.transpose(1, 2)

        k_raw = self.k_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim)
        v_raw = (
            k_raw
            if self.use_v_from_k
            else self.v_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim)
        )

        k = self.k_norm(k_raw).transpose(1, 2)
        v = _rms_norm_no_scale(v_raw, self.rms_eps).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
        else:
            cache_offset = 0

        n_rep = self.n_head // self.n_head_kv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        if kv_cache is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset, self.sliding_window)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        elif self.sliding_window is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset, self.sliding_window)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class Gemma4DecoderLayer(nn.Module):
    """Gemma's real "sandwich" norm layout (confirmed via the real HF

    `Gemma4TextDecoderLayer.forward`, not assumed from general Gemma
    familiarity): a pre-norm AND a post-norm around *each* sub-block
    (attention, then MLP) - four RMSNorms per layer, not the usual two -
    plus a single learned scalar (`layer_scalar`/`layer_output_scale` in
    the real GGUF) multiplying the *entire* layer's output at the very
    end, after both sub-blocks and their residual adds.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        ffn_len: int,
        rms_eps: float,
        use_v_from_k: bool,
        sliding_window: int | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = Gemma4Attention(
            n_embd, n_head, n_head_kv, head_dim, rms_eps, use_v_from_k, sliding_window, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mlp = Gemma4MLP(n_embd, ffn_len, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        stage_started = time.monotonic()
        residual = x
        hidden = self.input_layernorm(x)
        hidden = self.self_attn(hidden, cos, sin, kv_cache, layer_idx)
        hidden = self.post_attention_layernorm(hidden)
        x = residual + hidden
        logger.debug(
            "  layer %s: attention %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )

        stage_started = time.monotonic()
        residual = x
        hidden = self.pre_feedforward_layernorm(x)
        hidden = self.mlp(hidden)
        hidden = self.post_feedforward_layernorm(hidden)
        x = residual + hidden
        logger.debug(
            "  layer %s: ffn %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )

        return x * self.layer_scalar
