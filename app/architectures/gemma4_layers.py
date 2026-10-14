import logging
import time

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.fused_attention import fused_cached_attention
from app.architectures.layers import RMSNorm
from app.architectures.rope import apply_rotary_pos_emb
from app.architectures.shared_projection import project_shared
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


def _causal_mask(
    q_len: int,
    kv_len: int,
    cache_offset: int,
    window: int | None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Same causal rule as Mistral3DecoderLayer's `_causal_mask`, plus an optional sliding window:
    query i attends to
    key j iff `j <= i + cache_offset` (causal) and, if `window` is set, `j > i + cache_offset -
    window` (Gemma4's local layers).
    """
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    causal = kv_idx <= (q_idx + cache_offset)
    if window is None:
        return causal
    return causal & (kv_idx > (q_idx + cache_offset - window))


def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma4's `v_norm`: an active step with no learnable weight (`with_scale=False` in HF
    `Gemma4RMSNorm`; no
    `attn_v_norm.weight` in the GGUF), so a plain function rather than a module with nothing to
    load.
    """
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(input_dtype)


class Gemma4MLP(nn.Module):
    """Gated FFN with GELU-tanh (HF default `hidden_activation="gelu_pytorch_tanh"`; NOT SiLU), so
    it can't reuse the
    shared `SwiGLUMLP` despite the same gate/up/down shape.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = project_shared(self, ("gate_proj", "up_proj"), x)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)


class Gemma4Attention(nn.Module):
    """QK-norm GQA attention with per-layer-type head_dim/kv-heads (local/sliding layers: 8 kv-heads
    of dim 256; global: 1 kv-head, effectively MQA, of dim 512; from this GGUF's per-layer tensor
    shapes).

    `use_v_from_k` (HF `attention_k_eq_v`) reuses the raw (pre-`k_norm`, pre-RoPE) key projection as
    the value input instead of a `v_proj`. Per-layer tensor presence decides it
    (`Gemma4Architecture.from_gguf` checks `attn_v.weight` per layer), not the layer type: a
    `gemma-4-E2B-it` GGUF has `attn_v.weight` on every layer.

    `has_own_kv` (HF `num_kv_shared_layers`/llama.cpp `n_layer_kv_from_start`): the last
    `n_kv_shared_layers` layers have no k/v projections and use `shared_kv`, a `(k, v)` pair from an
    earlier layer in this pass. `Gemma4Architecture._forward_impl` picks the provider (llama.cpp's
    `n_layer_kv_from_start - (2 if swa else 1)`, `src/llama-model.cpp`) and threads it through.

    `scaling=1.0` (not `1/sqrt(head_dim)`) is a Gemma4 choice: QK-norm already controls query/key
    magnitudes.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        rms_eps: float,
        has_own_kv: bool,
        use_v_from_k: bool,
        sliding_window: int | None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.rms_eps = rms_eps
        self.has_own_kv = has_own_kv
        self.use_v_from_k = use_v_from_k
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)
        if has_own_kv:
            self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
            self.v_proj = (
                None
                if use_v_from_k
                else nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
            )
            self.k_norm = RMSNorm(head_dim, rms_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = x.shape
        cache_offset = kv_cache.length if kv_cache is not None else 0

        q = self.q_norm(self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim))
        q = q.transpose(1, 2)
        q, _ = apply_rotary_pos_emb(q, q, cos, sin)

        kv_out = None
        if self.has_own_kv:
            k_raw = self.k_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim)
            v_raw = (
                k_raw
                if self.use_v_from_k
                else self.v_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim)
            )

            k = self.k_norm(k_raw).transpose(1, 2)
            v = rms_norm_no_scale(v_raw, self.rms_eps).transpose(1, 2)
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)

            if kv_cache is not None:
                k, v = kv_cache.update(layer_idx, k, v)
            kv_out = (k, v)
        else:
            k, v = shared_kv

        # enable_gqa: SDPA shares each k/v head across its query-head group itself, with no
        # repeat_interleave copy of the
        # cached k/v per step (measured on the dev machine: at 200 tokens that copy cost ~90 ms
        # per generated token over 28 layers vs ~7 ms without; the gap grows with the conversation).
        if kv_cache is not None or self.sliding_window is not None:
            out = fused_cached_attention(q, k, v, cache_offset, 1.0, self.sliding_window)
            if out is None:
                mask = _causal_mask(
                    seq_len, k.shape[-2], cache_offset, self.sliding_window, q.device
                )
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, scale=1.0, enable_gqa=True
                )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=1.0, enable_gqa=True
            )
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out), kv_out


class Gemma4DecoderLayer(nn.Module):
    """Gemma's "sandwich" norm layout (HF `Gemma4TextDecoderLayer.forward`): a pre-norm and
    post-norm around each sub-block (attention, MLP), four RMSNorms per layer. `per_layer_dim > 0`
    (Per-Layer Embeddings, `gemma4_ple.py`) adds a third sub-block after the FFN's residual add:
    gate -> gelu-tanh -> multiply by this layer's PLE slice -> project up -> norm -> residual add
    (op order checked against HF `modular_gemma4.py` and llama.cpp's `src/models/gemma4.cpp`).
    `layer_scalar`/`layer_output_scale` multiplies the entire layer output last, after the PLE
    sub-block (same two sources)."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        ffn_len: int,
        rms_eps: float,
        has_own_kv: bool,
        use_v_from_k: bool,
        sliding_window: int | None,
        per_layer_dim: int,
        moe: nn.Module | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = Gemma4Attention(
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            rms_eps,
            has_own_kv,
            use_v_from_k,
            sliding_window,
            dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.pre_feedforward_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mlp = Gemma4MLP(n_embd, ffn_len, dtype=dtype)
        self.post_feedforward_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.moe = moe
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype))

        self.per_layer_dim = per_layer_dim
        if per_layer_dim:
            self.per_layer_input_gate = nn.Linear(n_embd, per_layer_dim, bias=False, dtype=dtype)
            self.per_layer_projection = nn.Linear(per_layer_dim, n_embd, bias=False, dtype=dtype)
            self.post_per_layer_input_norm = RMSNorm(n_embd, rms_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: KVCache | None = None,
        layer_idx: int | None = None,
        shared_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        per_layer_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        stage_started = time.monotonic()
        residual = x
        hidden = self.input_layernorm(x)
        hidden, kv_out = self.self_attn(hidden, cos, sin, kv_cache, layer_idx, shared_kv)
        hidden = self.post_attention_layernorm(hidden)
        x = residual + hidden
        logger.debug(
            "  layer %s: attention %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000
        )

        stage_started = time.monotonic()
        residual = x
        hidden = self.pre_feedforward_layernorm(x)
        hidden = self.mlp(hidden)
        if self.moe is not None:
            hidden = self.moe(hidden, residual)
        hidden = self.post_feedforward_layernorm(hidden)
        x = residual + hidden
        logger.debug("  layer %s: ffn %.1fms", layer_idx, (time.monotonic() - stage_started) * 1000)

        if self.per_layer_dim:
            residual = x
            hidden = F.gelu(self.per_layer_input_gate(x), approximate="tanh")
            hidden = hidden * per_layer_input
            hidden = self.per_layer_projection(hidden)
            hidden = self.post_per_layer_input_norm(hidden)
            x = residual + hidden

        return x * self.layer_scalar, kv_out
