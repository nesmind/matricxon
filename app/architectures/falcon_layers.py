import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class FalconMLP(nn.Module):
    """`down(gelu(up(x)))` - a plain, non-gated, **bias-free** GELU MLP: real `tiiuae/falcon-7b`

    (`config.bias = False`) - confirmed via a real downloaded GGUF header: no `ffn_up.bias`/
    `ffn_down.bias` tensor at all. Exact GELU (HF's own default `config.activation = "gelu"`,
    not the tanh approximation `Starcoder2MLP` uses) - same formula as `Phi2MLP`, just bias-free,
    so its own small class rather than adding a third flag to that one.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class FalconDecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))` - one shared, real **biased**

    `nn.LayerNorm` feeds both branches (confirmed via a real `tiiuae/falcon-7b` GGUF: real
    `attn_norm.bias` tensor present, and no `attn_norm_2` tensor at all - this is the plain
    `parallel_attn=True, new_decoder_architecture=False` real Falcon variant; the newer dual-norm
    `new_decoder_architecture` variant and the older `parallel_attn=False`/ALiBi `falcon-rw-*`
    variant are both out of scope for this pass, not silently assumed to be the same shape).

    `self_attn` reuses `QwenAttention` directly (`qkv_bias=False, qk_norm_eps=None,
    o_proj_bias=False`) - real Falcon-7B's attention (real MQA, `head_count_kv=1`; full RoPE, no
    bias anywhere) is exactly what those flags already produce, `n_head_kv` already generalizing
    to MQA with no special-casing needed. The real fused `attn_qkv` GGUF tensor still needs
    splitting into this class's separate q/k/v projections - see `FalconArchitecture.
    _materialize_weights`'s own comment for the real MQA-aware (not equal-thirds) split.
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
