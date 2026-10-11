from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.fused_attention import fused_cached_attention
from app.architectures.layers import RMSNorm
from app.architectures.nemotron_h_mamba2 import NemotronHMamba2Mixer
from app.gguf.loader import GGUFModelLoader


def _causal_mask(
    q_len: int, kv_len: int, cache_offset: int, device: torch.device | None = None
) -> torch.Tensor:
    q_idx = torch.arange(q_len, device=device).unsqueeze(1)
    kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class NemotronHAttention(nn.Module):
    """Plain GQA with no RoPE at all: no rotary tensors in the GGUF, no rotary path in
    `modeling_nemotron_h.py`, `llama_model_rope_type` returns `LLAMA_ROPE_TYPE_NONE`, and the
    GGUF's `context_length=2^20` is a side effect of the converter's
    `add_rope_scaling_finetuned(use_rope=False)`. Default SDPA scale (`head_dim**-0.5`); the
    Granite-style multiplier is `nemotron_h_moe`-only."""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.q_proj = nn.Linear(n_embd, n_head * head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(n_embd, n_head_kv * head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(n_head * head_dim, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_head_kv, self.head_dim).transpose(1, 2)
        # No apply_rotary_pos_emb call: the one structural difference from
        # GroupedQueryAttention/GraniteAttention.

        if hybrid_cache is not None:
            cache_offset = hybrid_cache.length
            k, v = hybrid_cache.update_attention(layer_idx, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)
        else:
            cache_offset = 0

        # enable_gqa: SDPA shares each k/v head across its query-head group itself, with no
        # repeat_interleave copy of the cached
        # k/v per step (measured on the dev machine: at 200 tokens that copy cost ~90 ms per
        # generated token over 28 layers vs ~7 ms without; the gap grows with the conversation).
        if hybrid_cache is not None:
            out = fused_cached_attention(q, k, v, cache_offset)
            if out is None:
                mask = _causal_mask(seq_len, k.shape[-2], cache_offset, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class NemotronHMLP(nn.Module):
    """`down_proj(relu(up_proj(x))**2)`: non-gated, squared-ReLU (`mlp_hidden_act = "relu2"` in
    `modeling_nemotron_h.py`); the
    GGUF has no `ffn_gate`, so this is a new simpler class, not the shared gated `SwiGLUMLP`.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).pow(2))


class NemotronHMambaBlock(nn.Module):
    """`x + mixer(input_layernorm(x))`: the same single shared pre-norm every layer type uses
    (`attn_norm.weight` is the one
    tensor name shared by Mamba/attention/MLP layers; no post-norm anywhere).
    """

    def __init__(
        self,
        n_embd: int,
        rms_eps: float,
        mixer: NemotronHMamba2Mixer,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mixer = mixer

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        return x + self.mixer(self.input_layernorm(x), hybrid_cache, layer_idx)


class NemotronHAttentionBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        rms_eps: float,
        self_attn: NemotronHAttention,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.self_attn = self_attn

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        return x + self.self_attn(self.input_layernorm(x), hybrid_cache, layer_idx)


class NemotronHMLPBlock(nn.Module):
    def __init__(
        self, n_embd: int, rms_eps: float, mlp: NemotronHMLP, dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(n_embd, rms_eps, dtype=dtype)
        self.mlp = mlp

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        del hybrid_cache, layer_idx
        return x + self.mlp(self.input_layernorm(x))


def build_nemotron_h_layer(
    layer_type: str,
    n_embd: int,
    n_head: int,
    n_head_kv: int,
    head_dim: int,
    ffn_len: int,
    rms_eps: float,
    mamba_d_inner: int,
    mamba_num_heads: int,
    mamba_head_dim: int,
    d_state: int,
    n_group: int,
    conv_kernel: int,
    dtype: torch.dtype,
) -> nn.Module:
    """One layer's construction, dispatched by its per-index type (`NemotronHArchitecture.__init__`
    derives `layer_types` from the
    two GGUF per-layer arrays); kept beside the layer classes so the architecture file stays on
    metadata parsing/materialization.
    """
    if layer_type == "mamba":
        mixer = NemotronHMamba2Mixer(
            n_embd,
            mamba_d_inner,
            mamba_num_heads,
            mamba_head_dim,
            d_state,
            n_group,
            conv_kernel,
            rms_eps,
            dtype=dtype,
        )
        return NemotronHMambaBlock(n_embd, rms_eps, mixer, dtype=dtype)
    if layer_type == "attention":
        attn = NemotronHAttention(n_embd, n_head, n_head_kv, head_dim, dtype=dtype)
        return NemotronHAttentionBlock(n_embd, rms_eps, attn, dtype=dtype)
    mlp = NemotronHMLP(n_embd, ffn_len, dtype=dtype)
    return NemotronHMLPBlock(n_embd, rms_eps, mlp, dtype=dtype)


def materialize_mamba_layer(
    loader: GGUFModelLoader,
    prefix: str,
    mixer: NemotronHMamba2Mixer,
    load_projection: Callable[..., nn.Module],
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    """`ssm_in`/`ssm_out` are plain 2D `nn.Linear` weights (99.9% of a checkpoint's SSM elements),
    so they go through `_load_projection` like other architectures' projections.
    The other SSM tensors aren't 2D Linear-shaped and use a plain `.copy_` into their
    `nn.Parameter` whatever `enabled`."""
    mixer.in_proj = load_projection(loader, prefix + "ssm_in.weight", mixer.in_proj, dtype, enabled)
    mixer.conv1d_weight.copy_(loader.load_tensor(prefix + "ssm_conv1d.weight"))
    mixer.conv1d_bias.copy_(loader.load_tensor(prefix + "ssm_conv1d.bias"))
    mixer.dt_bias.copy_(loader.load_tensor(prefix + "ssm_dt.bias"))
    mixer.a.copy_(loader.load_tensor(prefix + "ssm_a").reshape(-1))
    mixer.d.copy_(loader.load_tensor(prefix + "ssm_d").reshape(-1))
    mixer.norm_weight.copy_(loader.load_tensor(prefix + "ssm_norm.weight"))
    mixer.out_proj = load_projection(
        loader, prefix + "ssm_out.weight", mixer.out_proj, dtype, enabled
    )


def materialize_attention_layer(
    loader: GGUFModelLoader,
    prefix: str,
    attn: NemotronHAttention,
    load_projection: Callable[..., nn.Module],
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    # No unpermute_rope_rows: that permutation exists only for rotate-half RoPE, which this
    # architecture never applies.
    attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
    attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
    attn.v_proj = load_projection(loader, prefix + "attn_v.weight", attn.v_proj, dtype, enabled)
    attn.o_proj = load_projection(
        loader, prefix + "attn_output.weight", attn.o_proj, dtype, enabled
    )


def materialize_mlp_layer(
    loader: GGUFModelLoader,
    prefix: str,
    mlp: NemotronHMLP,
    load_projection: Callable[..., nn.Module],
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    mlp.up_proj = load_projection(loader, prefix + "ffn_up.weight", mlp.up_proj, dtype, enabled)
    mlp.down_proj = load_projection(
        loader, prefix + "ffn_down.weight", mlp.down_proj, dtype, enabled
    )
