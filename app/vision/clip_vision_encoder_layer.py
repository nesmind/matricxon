import torch
import torch.nn.functional as F
from torch import nn


class ClipVisionEncoderLayer(nn.Module):
    """A standard pre-norm ViT block: plain (non-causal, non-grouped) multi-head self-attention with
    biases on every projection (unlike matricxon's bias-free text decoders), then a 2-layer GELU
    MLP. Pre-norm (LN before each sub-block), unlike `BertArchitecture`'s post-norm: `ln1`/`ln2`
    feed into their sub-block.

    `fc1`/`fc2` (expand then contract) are named generically, not `ffn_up`/`ffn_down`: by the
    tensors' bias lengths, this GGUF's `ffn_up.weight`/`ffn_up.bias` is the contracting projection
    (bias has `n_embd` elements) and `ffn_down` the expanding one (`ffn_len` elements), the opposite
    of the names. `ClipVisionEncoder.from_gguf` shows which tensor feeds which.
    """

    def __init__(
        self, n_embd: int, n_head: int, ffn_len: int, eps: float, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd, eps=eps, dtype=dtype)
        self.q_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.out_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.ln2 = nn.LayerNorm(n_embd, eps=eps, dtype=dtype)
        self.fc1 = nn.Linear(n_embd, ffn_len, dtype=dtype)
        self.fc2 = nn.Linear(ffn_len, n_embd, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape

        residual = x
        h = self.ln1(x)
        q = self.q_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, n_embd)
        x = residual + self.out_proj(attn)

        residual = x
        h = self.ln2(x)
        h = self.fc2(F.gelu(self.fc1(h)))
        return residual + h
