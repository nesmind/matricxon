import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.shared_projection import project_shared
from app.native.fused_ops import FusedOps


def unpermute_rope_rows(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """Reverses the row permutation llama.cpp's HF->GGUF converter applies to attn_q/attn_k, for the
    families that need it: `mistral3`/`llama`. Not universal: `gemma4` needs none, and oracle
    validation showed `qwen2`/`qwen3` need none (applying it made the model generate
    garbage, never predicting "Paris" after "The capital of France is", though every other check
    passed). Verify per architecture against real weights; don't infer from rotate-half resemblance,
    which was already wrong once for Qwen.

    Where needed, the converter interleaves each head's rotary pairs into ggml's split-half
    convention by swapping a (2, head_dim/2) grouping to (head_dim/2, 2), while
    `apply_rotary_pos_emb` here (and every HF Mistral/Llama-family model) expects natural row order.
    Skipping it where needed makes attention silently wrong (against an HF oracle: weight cosine
    ~0.03 without, ~0.997 with, for `mistral3`/`llama`).
    """
    dim0 = weight.shape[0]
    head_dim = dim0 // n_head
    return (
        weight.reshape(n_head, head_dim // 2, 2, *weight.shape[1:])
        .transpose(1, 2)
        .reshape(weight.shape)
    )


class SwiGLUMLP(nn.Module):
    """Gated (SwiGLU) feed-forward `down(silu(gate(x)) * up(x))`, shared by every architecture using
    this block (mistral3's
    decoder MLP, nomic-bert's FFN: its config has `activation_function: "swiglu"` and a `ffn_gate`
    GGUF tensor beside
    `ffn_up`/`ffn_down`). `bias` defaults to `False` (every user so far; nomic-bert's config has
    `mlp_fc1_bias`/`mlp_fc2_bias: false`).
    """

    def __init__(
        self,
        n_embd: int,
        ffn_len: int,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, ffn_len, bias=bias, dtype=dtype)
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=bias, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=bias, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = project_shared(self, ("gate_proj", "up_proj"), x)
        fused = FusedOps.active()
        hidden = fused.silu_mul(gate, up) if fused is not None else None
        return self.down_proj(hidden if hidden is not None else F.silu(gate) * up)


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight, computed in float32 whatever the parameter dtype: the
    mean-square reduction
    is precision-sensitive and every reference Llama-family implementation does this (bf16
    activations lose too much accuracy).
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = FusedOps.active()
        fused = ops.rms_norm(x, self.weight, self.eps) if ops is not None else None
        if fused is not None:
            return fused
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x.to(input_dtype)) * self.weight
