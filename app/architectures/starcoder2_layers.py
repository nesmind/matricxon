import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class Starcoder2MLP(nn.Module):
    """`down(gelu_tanh(up(x)))` - a plain, non-gated MLP with a real bias on both projections

    (`config.use_bias`, real default `True` on every released checkpoint - confirmed via HF's own
    `modeling_starcoder2.py`'s `Starcoder2MLP`), using the **tanh approximation** of GELU
    (`hidden_act = "gelu_pytorch_tanh"`), not the exact/erf-based GELU `Phi2MLP` uses - a real,
    confirmed difference between the two plain-GELU-MLP architectures in this repo, not
    interchangeable.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=True, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class Starcoder2DecoderLayer(nn.Module):
    """Same sequential pre-norm two-block layout as `QwenDecoderLayer` (real bias-free `RMSNorm`

    reused everywhere else in this repo) - but StarCoder2's real norm is a plain, biased
    `nn.LayerNorm` instead (confirmed: HF's own `modeling_starcoder2.py` constructs
    `nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)` with no `bias=False` override,
    and the real GGUF metadata key itself signals this - `attention.layer_norm_epsilon`, the same
    key `command_r.py` uses for its own bias-free `LayerNorm`, not the `...rms_epsilon` key every
    RMSNorm architecture here uses). `self_attn` reuses `QwenAttention` with `o_proj_bias=True`
    (see that class's own docstring) rather than a new attention class - StarCoder2's real shape
    (GQA, full RoPE, bias on every one of q/k/v/o, no QK-norm) is exactly what `qkv_bias=True,
    qk_norm_eps=None, o_proj_bias=True` already produces.
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
