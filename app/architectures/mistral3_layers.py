import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.fused_attention import fused_cached_attention
from app.architectures.layers import RMSNorm, SwiGLUMLP  # noqa: F401 - re-exported, see mistral3.py
from app.architectures.rope import apply_rotary_pos_emb
from app.architectures.shared_projection import project_shared
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(
    q_len: int, kv_len: int, cache_offset: int, device: torch.device | None = None
) -> torch.Tensor:
    """Query i (local index) may attend to key j iff j <= i + cache_offset: its own absolute
    position and everything
    before. `cache_offset=0` and `q_len == kv_len` is plain causal masking (prefill); a one-token
    decode step (`q_len=1`) attends to the whole cache plus itself.
    """
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
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

        q, k, v = project_shared(self, ("q_proj", "k_proj", "v_proj"), x)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            cache_offset = kv_cache.length
            k, v = kv_cache.update(layer_idx, k, v)
            # KVCache stores every layer in one dtype decided by its caller (ChatEngine.stream):
            # fine with one model-wide dtype, but
            # under mixed per-layer dtypes (ModelManager._load) it can differ from *this* layer's
            # q/k/v dtype.
            # scaled_dot_product_attention needs one dtype for q/k/v (it doesn't promote like
            # RMSNorm's elementwise ops): cast to q's, as apply_rotary_pos_emb does for cos/sin.
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        else:
            cache_offset = 0

        # enable_gqa: SDPA shares each k/v head across its query-head group itself, with no
        # repeat_interleave copy of the cached
        # k/v per step (measured on the dev machine: at 200 tokens that copy cost ~90 ms per
        # generated token over 28 layers vs ~7 ms without; the gap grows with the conversation).
        if kv_cache is not None:
            out = fused_cached_attention(q, k, v, cache_offset)
            if out is None:
                mask = _causal_mask(seq_len, k.shape[-2], cache_offset, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class Mistral3DecoderLayer(nn.Module):
    """Same pre-norm two-block layout as `GraniteDecoderLayer` (`granite_layers.py`). `mlp` is
    injected by the caller: dense `Mistral3TextArchitecture`/`LlamaArchitecture` inject a
    `SwiGLUMLP`, and a Mixtral-style `llama`-arch MoE checkpoint injects a `GraniteMoeFFN`
    (`llama.py`). The attention block is written and tested once for both."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        rms_eps: float,
        mlp: nn.Module,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = GroupedQueryAttention(n_embd, n_head, n_head_kv, head_dim, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mlp = mlp

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
        logger.debug("  layer %s: ffn %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000)
        return x
