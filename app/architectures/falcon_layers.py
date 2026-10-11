import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class FalconMLP(nn.Module):
    """`down(gelu(up(x)))`: a plain, non-gated, **bias-free** GELU MLP (`tiiuae/falcon-7b`,
    `config.bias = False`; a
    real GGUF header has no `ffn_up.bias`/`ffn_down.bias`). Exact GELU (HF default
    `config.activation = "gelu"`, not
    the tanh approximation of `Starcoder2MLP`). Same formula as `Phi2MLP` but bias-free, hence its
    own class rather than a third flag there.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class FalconDecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))`: one shared biased `nn.LayerNorm`
    feeds both branches (a real `tiiuae/falcon-7b` GGUF has `attn_norm.bias` and no `attn_norm_2`:
    the plain `parallel_attn=True, new_decoder_architecture=False` variant; the dual-norm
    `new_decoder_architecture` and ALiBi `falcon-rw-*` variants are out of scope).

    `self_attn` reuses `QwenAttention` (`qkv_bias=False, qk_norm_eps=None, o_proj_bias=False`):
    Falcon-7B's attention (MQA, `head_count_kv=1`; full RoPE, no bias) is what those flags produce,
    `n_head_kv` already generalizing to MQA. The fused `attn_qkv` tensor is split into this class's
    q/k/v projections; see `FalconArchitecture._materialize_weights` for the MQA-aware (not
    equal-thirds) split.
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
            n_embd, n_head, n_head_kv, head_dim, qkv_bias=False, qk_norm_eps=None, dtype=dtype
        )
        self.mlp = FalconMLP(n_embd, ffn_len, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
    ) -> torch.Tensor:
        h = self.input_layernorm(x)
        return x + self.self_attn(h, cos, sin, kv_cache, layer_idx) + self.mlp(h)
