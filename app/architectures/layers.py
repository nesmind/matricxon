import torch
import torch.nn.functional as F
from torch import nn


def unpermute_rope_rows(weight: torch.Tensor, n_head: int) -> torch.Tensor:
    """Reverses the row permutation llama.cpp's HF->GGUF converter applies to

    attn_q/attn_k weights for the model families that need it - confirmed
    real for `mistral3`/`llama` (first written for `Mistral3TextArchitecture`,
    confirmed to also apply to plain `llama` models - see M10's ROADMAP
    entry), not for any one of matricxon's architectures universally: `gemma4`
    needs no permutation at all (see that architecture's own comment), and
    real-weight oracle validation confirmed (2026-09-22) `qwen2`/`qwen3` need
    none either - applying this to real Qwen weights silently turns a working
    model into one that only ever generates garbage (confirmed live: never
    predicted "Paris" after "The capital of France is", despite every other
    check passing cleanly). Whether a given real architecture needs this at
    all must be verified per-architecture against real weights, not assumed
    from the "it's a rotate-half model" family resemblance alone - that
    assumption was already wrong once for Qwen.

    Where it IS needed: that converter interleaves each head's rotary pairs
    into ggml's split-half convention by swapping a (2, head_dim/2) grouping
    to (head_dim/2, 2) per head, so GGUF's attn_q.weight/attn_k.weight ship
    already permuted - while `apply_rotary_pos_emb` here (and in every HF
    Mistral/Llama-family model) expects the un-permuted, "natural" row
    order. Skipping this where it IS needed turns attention silently wrong
    rather than raising (confirmed against an HF safetensors oracle: raw
    weight cosine similarity was ~0.03 without this, ~0.997 - full
    quantization-noise level - with it, for `mistral3`/`llama`).
    """
    dim0 = weight.shape[0]
    head_dim = dim0 // n_head
    return (
        weight.reshape(n_head, head_dim // 2, 2, *weight.shape[1:])
        .transpose(1, 2)
        .reshape(weight.shape)
    )


class SwiGLUMLP(nn.Module):
    """Gated (SwiGLU) feed-forward: `down(silu(gate(x)) * up(x))` - shared by
    every architecture that uses this exact block (mistral3's decoder MLP,
    nomic-bert's FFN - confirmed via its real config's
    `activation_function: "swiglu"` and a real `ffn_gate` GGUF tensor
    alongside `ffn_up`/`ffn_down`), so it lives here rather than duplicated
    per architecture. `bias` defaults to `False` (matches every real user of
    this block so far); nomic-bert's real config also confirms
    `mlp_fc1_bias`/`mlp_fc2_bias: false`.
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
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * weight - computed in float32 regardless of
    the parameter dtype, since the mean-square reduction is precision-
    sensitive and this is how every reference Llama-family implementation
    does it (bf16 activations would otherwise lose too much accuracy here).
    """

    def __init__(self, dim: int, eps: float, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (x.to(input_dtype)) * self.weight
