import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import SwiGLUMLP
from app.architectures.rope import apply_rotary_pos_emb

logger = logging.getLogger(__name__)


class NomicBertSelfAttention(nn.Module):
    """Fused QKV projection - confirmed against the real
    `nomic-embed-text-v1.5` GGUF: a single `attn_qkv.weight` tensor, not
    separate `attn_q`/`attn_k`/`attn_v` ones. Split as `[q; k; v]`, each of
    width `n_embd` (verified via the HF oracle cross-check, not assumed -
    see scripts/oracle/validate_nomic_bert.py).

    No bias anywhere (the real config confirms `qkv_proj_bias: false`), and
    rotary position embeddings applied with the plain split-half
    `rotate_half` (the config's `rotary_emb_interleaved: false` means no
    row-permutation trick is needed here, unlike mistral3's GGUF weights).
    """

    def __init__(
        self, n_embd: int, n_head: int, head_dim: int, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(n_embd, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(n_embd, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Non-causal, no padding mask - see BertSelfAttention's note.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, n_embd)
        return self.o_proj(out)


class NomicBertEncoderLayer(nn.Module):
    """Post-LN, same layer structure as classic BERT (`prenorm: false` in
    the real config) but with the fused-QKV/RoPE attention and SwiGLU FFN
    above instead of BERT's separate-QKV/GELU."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        head_dim: int,
        ffn_len: int,
        layer_norm_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.self_attn = NomicBertSelfAttention(n_embd, n_head, head_dim, dtype=dtype)
        self.attn_output_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)
        self.mlp = SwiGLUMLP(n_embd, ffn_len, dtype=dtype)
        self.layer_output_norm = nn.LayerNorm(n_embd, eps=layer_norm_eps, dtype=dtype)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        stage_started = time.monotonic()
        x = self.attn_output_norm(x + self.self_attn(x, cos, sin))
        logger.debug("  attention %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.layer_output_norm(x + self.mlp(x))
        logger.debug("  ffn %.1fms", (time.monotonic() - stage_started) * 1000)
        return x
