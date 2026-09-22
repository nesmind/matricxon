import torch
from torch import nn

from app.architectures.layers import SwiGLUMLP
from app.architectures.qwen_layers import QwenAttention
from app.runtime.kv_cache import KVCache


class CommandRDecoderLayer(nn.Module):
    """Parallel residual: `x = x + attn(norm(x)) + mlp(norm(x))` - one shared norm feeds *both*
    branches from the same input, added back together (confirmed exact formula from HF's real
    `modeling_cohere.py`). Structurally identical in shape to `Phi2DecoderLayer`
    (`app/architectures/phi2_layers.py`) - the only precedent for a parallel-residual block in
    this repo - only the norm type, attention shape, and MLP shape differ.

    `input_layernorm` is real, bias-free `nn.LayerNorm` (mean-centered, unlike every RMSNorm-based
    architecture here so far - confirmed from HF's own `CohereLayerNorm` source), not `RMSNorm`.

    `self_attn` reuses `QwenAttention` directly (`app/architectures/qwen_layers.py`) with
    `qkv_bias=False, qk_norm_eps=None` - Command-R's real attention shape (GQA, full RoPE, no
    bias, no QK-norm at this scale - QK-norm only exists on the 104B Command-R+, out of scope) is
    exactly what those two flags already produce, so there's no need for a new attention class -
    the same cross-architecture reuse precedent `llama.py` already set by reusing
    `mistral3_layers.py`'s `GroupedQueryAttention` directly. `mlp` is a plain shared `SwiGLUMLP`
    (confirmed real GGUF tensor names `ffn_gate`/`ffn_up`/`ffn_down` - gated, not Phi-2's plain
    GELU MLP).
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
