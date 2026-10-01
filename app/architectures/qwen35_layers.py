from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm, SwiGLUMLP
from app.architectures.qwen35_deltanet import Qwen35GatedDeltaNet
from app.architectures.qwen_layers import _causal_mask
from app.architectures.rope import apply_rotary_pos_emb_partial
from app.gguf.loader import GGUFModelLoader

LoadProjection = Callable[..., nn.Module]


class Qwen35Attention(nn.Module):
    """Qwen3.5's full-attention layer (every 4th): Qwen3-style GQA with QK-norm, plus two deltas -
    `attn_q` is twice as wide (per head, `[query | gate]` interleaved) and the attention output
    is multiplied by `sigmoid(gate)` before `o_proj`; and RoPE only rotates the first `rope_dim`
    of each head's dims (text-only M-RoPE collapses to plain NEOX RoPE: all three position
    streams carry the same index).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim * 2, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)
        self.k_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hybrid_cache: object,
        layer_idx: int,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q_and_gate = self.q_proj(x).view(batch, seq_len, self.n_head, 2 * self.head_dim)
        q, gate = q_and_gate.chunk(2, dim=-1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb_partial(q, k, cos, sin)

        if hybrid_cache is not None:
            cache_offset = hybrid_cache.length
            k, v = hybrid_cache.update_attention(layer_idx, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2) * torch.sigmoid(gate)
        return self.o_proj(out.reshape(batch, seq_len, self.n_head * self.head_dim))


class Qwen35DecoderLayer(nn.Module):
    """Pre-norm two-block layer; the token mixer is either `Qwen35Attention` (`self_attn`) or
    `Qwen35GatedDeltaNet` (`linear_attn`) - the FFN is the same dense SwiGLU in both."""

    def __init__(
        self,
        n_embd: int,
        ffn_len: int,
        rms_eps: float,
        attention: Qwen35Attention | None,
        linear_attn: Qwen35GatedDeltaNet | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.post_attention_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = attention
        self.linear_attn = linear_attn
        self.mlp = SwiGLUMLP(n_embd, ffn_len, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        hybrid_cache: object,
        layer_idx: int,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        if self.self_attn is not None:
            x = x + self.self_attn(h, cos, sin, hybrid_cache, layer_idx)
        else:
            x = x + self.linear_attn(h, hybrid_cache, layer_idx)
        return x + self.mlp(self.post_attention_layernorm(x))


def materialize_decoder_layer(
    loader: GGUFModelLoader,
    prefix: str,
    layer: Qwen35DecoderLayer,
    load_projection: LoadProjection,
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    """Copies one layer's tensors in. No `unpermute_rope_rows` anywhere (same reasoning as
    `Qwen3Architecture`: Qwen's GGUF q/k rows are already in the order its RoPE expects)."""

    def proj(suffix: str, target: nn.Linear) -> nn.Module:
        return load_projection(loader, prefix + suffix, target, dtype, enabled)

    layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
    layer.post_attention_layernorm.weight.copy_(
        loader.load_tensor(prefix + "post_attention_norm.weight")
    )
    layer.mlp.gate_proj = proj("ffn_gate.weight", layer.mlp.gate_proj)
    layer.mlp.up_proj = proj("ffn_up.weight", layer.mlp.up_proj)
    layer.mlp.down_proj = proj("ffn_down.weight", layer.mlp.down_proj)

    attn = layer.self_attn
    if attn is not None:
        attn.q_proj = proj("attn_q.weight", attn.q_proj)
        attn.k_proj = proj("attn_k.weight", attn.k_proj)
        attn.v_proj = proj("attn_v.weight", attn.v_proj)
        attn.o_proj = proj("attn_output.weight", attn.o_proj)
        attn.q_norm.weight.copy_(loader.load_tensor(prefix + "attn_q_norm.weight"))
        attn.k_norm.weight.copy_(loader.load_tensor(prefix + "attn_k_norm.weight"))
        return

    mixer = layer.linear_attn
    mixer.qkv_proj = proj("attn_qkv.weight", mixer.qkv_proj)
    mixer.z_proj = proj("attn_gate.weight", mixer.z_proj)
    mixer.beta_proj = proj("ssm_beta.weight", mixer.beta_proj)
    mixer.alpha_proj = proj("ssm_alpha.weight", mixer.alpha_proj)
    mixer.out_proj = proj("ssm_out.weight", mixer.out_proj)
    mixer.conv1d_weight.copy_(loader.load_tensor(prefix + "ssm_conv1d.weight"))
    mixer.dt_bias.copy_(loader.load_tensor(prefix + "ssm_dt.bias"))
    mixer.a.copy_(loader.load_tensor(prefix + "ssm_a").reshape(-1))
    mixer.norm_weight.copy_(loader.load_tensor(prefix + "ssm_norm.weight"))
