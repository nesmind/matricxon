import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm
from app.architectures.rope import apply_rotary_pos_emb
from app.runtime.kv_cache import KVCache


def _causal_mask(q_len: int, kv_len: int, cache_offset: int) -> torch.Tensor:
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class GraniteAttention(nn.Module):
    """`GroupedQueryAttention` (`mistral3_layers.py`) plus one real difference: `attention_scale`
    REPLACES `scaled_dot_product_attention`'s own default `1/sqrt(head_dim)` scale entirely -
    IBM Granite's real `attention_multiplier` (confirmed against `modeling_granite.py`: the
    attention-weight matmul is scaled by this value directly, nothing else). `attention_scale` is
    `None` when the real GGUF carries no `{arch}.attention.scale` key -
    `F.scaled_dot_product_attention(..., scale=None)` is PyTorch's own documented way of saying
    "use the default," so that fallback needs no extra branching here at all. Precedent for
    threading a non-default `scale=` through this call: `Gemma4Attention.forward`
    (`gemma4_layers.py`) already overrides it with a fixed `1.0`; this generalizes that to an
    arbitrary real per-model value.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        attention_scale: float | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.attention_scale = attention_scale
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
            k = k.to(q.dtype)
            v = v.to(q.dtype)
        else:
            cache_offset = 0

        n_rep = self.n_head // self.n_head_kv
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)

        if kv_cache is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.attention_scale
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.attention_scale
            )
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class GraniteDecoderLayer(nn.Module):
    """Same pre-norm two-block layout as `Mistral3DecoderLayer`, but the branch *output* (not the
    residual stream) is scaled by `residual_multiplier` before both adds - IBM Granite's real
    `residual_multiplier`, applied identically after attention and after the MLP (confirmed
    against `modeling_granite.py`: `hidden_states = residual + hidden_states * residual_multiplier`,
    same formula both places - not `(residual + hidden_states) * residual_multiplier`, a real,
    easy-to-get-backwards mistake this docstring exists to warn against).

    `mlp` is built by the caller, not this class (dependency injection) - dense
    `GraniteArchitecture` injects a plain `SwiGLUMLP` (`app/architectures/layers.py`),
    `GraniteMoeArchitecture` injects a `GraniteMoeFFN` (`granitemoe_layers.py`) - this class has
    zero MoE-awareness, so the attention block and both `residual_multiplier` sites are written
    and tested exactly once for both.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        rms_eps: float,
        residual_multiplier: float,
        attention_scale: float | None,
        mlp: nn.Module,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.residual_multiplier = residual_multiplier
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = GraniteAttention(
            n_embd, n_head, n_head_kv, head_dim, attention_scale, dtype=dtype
        )
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
        residual = x
        hidden = self.self_attn(self.input_layernorm(x), cos, sin, kv_cache, layer_idx)
        x = residual + hidden * self.residual_multiplier

        residual = x
        hidden = self.mlp(self.post_attention_layernorm(x))
        x = residual + hidden * self.residual_multiplier
        return x
