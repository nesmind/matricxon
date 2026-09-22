import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class BertSelfAttention(nn.Module):
    """Non-causal multi-head self-attention with separate Q/K/V projections
    and biases everywhere - classic BERT's attention block (confirmed
    against the real `all-MiniLM-L6-v2` GGUF: separate `attn_q`/`attn_k`/
    `attn_v` tensors, each with a bias).
    """

    def __init__(self, n_embd: int, n_head: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.q_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_embd, dtype=dtype)
        self.o_proj = nn.Linear(n_embd, n_embd, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # No causal mask, no padding mask - v1 encodes exactly one sequence
        # per /api/embeddings call (see EmbeddingEngine), so there's no
        # padding to mask out and every position may attend to every other.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class BertFFN(nn.Module):
    """Plain (non-gated) GELU feed-forward - the real `all-MiniLM-L6-v2`
    config's `hidden_act: "gelu"` is HF's exact erf-based GELU, not the
    tanh approximation some other configs mean by that name."""

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class BertEncoderLayer(nn.Module):
    """Post-LN: `x = LayerNorm(x + Attention(x))`, then
    `x = LayerNorm(x + FFN(x))` - classic BERT's layer structure, the
    opposite order from mistral3's pre-LN decoder layer."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        ffn_len: int,
        layer_norm_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.self_attn = BertSelfAttention(n_embd, n_head, dtype=dtype)
        self.attn_output_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)
        self.mlp = BertFFN(n_embd, ffn_len, dtype=dtype)
        self.layer_output_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stage_started = time.monotonic()
        x = self.attn_output_norm(x + self.self_attn(x))
        logger.debug("  attention %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.layer_output_norm(x + self.mlp(x))
        logger.debug("  ffn %.1fms", (time.monotonic() - stage_started) * 1000)
        return x
