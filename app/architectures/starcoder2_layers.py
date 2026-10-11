import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class Starcoder2MLP(nn.Module):
    """`down(gelu_tanh(up(x)))`: a plain non-gated MLP with a bias on both projections
    (`config.use_bias`, default `True` on every released
    checkpoint; HF's `Starcoder2MLP`), using the **tanh approximation** of GELU (`hidden_act =
    "gelu_pytorch_tanh"`), not the exact/erf GELU of
    `Phi2MLP`: the two plain-GELU-MLP architectures here are not interchangeable.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class Starcoder2DecoderLayer(nn.Module):
    """Same sequential pre-norm two-block layout as `QwenDecoderLayer`, but with a plain biased
    `nn.LayerNorm` (HF `modeling_starcoder2.py`: `nn.LayerNorm(config.hidden_size,
    eps=config.norm_epsilon)`; GGUF key `attention.layer_norm_epsilon`, as `command_r.py`, not the
    `...rms_epsilon` of RMSNorm architectures). `self_attn` reuses `QwenAttention` with
    `qkv_bias=True, qk_norm_eps=None, o_proj_bias=True`: StarCoder2's attention (GQA, full RoPE,
    bias on q/k/v/o, no QK-norm)."""

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
        self.input_layernorm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)
        self.self_attn = QwenAttention(
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            qkv_bias=True,
            qk_norm_eps=None,
            dtype=dtype,
            o_proj_bias=True,
        )
        self.post_attention_layernorm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)
        self.mlp = Starcoder2MLP(n_embd, ffn_len, dtype=dtype)

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
