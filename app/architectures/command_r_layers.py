import torch
from torch import nn

from app.architectures.layers import SwiGLUMLP
from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class CommandRDecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))`: one shared norm feeds both
    branches (formula from HF's `modeling_cohere.py`). Same shape as `Phi2DecoderLayer`
    (`phi2_layers.py`), the only other parallel-residual block here; only the norm type, attention
    shape and MLP shape differ.

    `input_layernorm` is a bias-free `nn.LayerNorm` (mean-centered, unlike the RMSNorm
    architectures; HF's `CohereLayerNorm`).

    `self_attn` reuses `QwenAttention` (`qwen_layers.py`) with `qkv_bias=False, qk_norm_eps=None`:
    Command-R's attention (GQA, full RoPE, no bias, no QK-norm at this scale), as `llama.py` reuses
    `mistral3_layers.py`'s `GroupedQueryAttention`. `mlp` is a shared `SwiGLUMLP` (GGUF
    `ffn_gate`/`ffn_up`/`ffn_down`: gated, not Phi-2's GELU MLP).
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        ffn_len: int,
        layer_norm_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(n_embd, eps=layer_norm_eps, bias=False, dtype=dtype)
        self.self_attn = QwenAttention(
            n_embd, n_head, n_head_kv, head_dim, qkv_bias=False, qk_norm_eps=None, dtype=dtype
        )
        self.mlp = SwiGLUMLP(n_embd, ffn_len, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, cos, sin, kv_cache, layer_idx)
        mlp_out = self.mlp(h)
        return x + attn_out + mlp_out
