from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm
from app.architectures.nemotron_h_mamba2 import NemotronHMamba2Mixer
from app.gguf.loader import GGUFModelLoader


def _causal_mask(q_len: int, kv_len: int, cache_offset: int) -> torch.Tensor:
    q_idx = torch.arange(q_len).unsqueeze(1)
    kv_idx = torch.arange(kv_len).unsqueeze(0)
    return kv_idx <= (q_idx + cache_offset)


class NemotronHAttention(nn.Module):
    """Plain GQA - **no RoPE at all** (confirmed multiple independent ways against the real HF
    source and llama.cpp's own C++ engine: no rotary tensors in the real GGUF file, no rotary
    code path in `modeling_nemotron_h.py`, `llama_model_rope_type()` explicitly returns
    `LLAMA_ROPE_TYPE_NONE` for this architecture, and the real GGUF's own
    `context_length=2^20` value is a direct side-effect of llama.cpp's converter calling
    `add_rope_scaling_finetuned(use_rope=False)` for this model family). Default
    `scaled_dot_product_attention` scale (`head_dim**-0.5`) - no Granite-style multiplier
    override for plain `nemotron_h` (that's `nemotron_h_moe`-only, confirmed via llama.cpp's own
    hparam-logging code, not relevant to the dense checkpoints this targets).
    """

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
        # No apply_rotary_pos_emb call - the one real structural difference from
        # GroupedQueryAttention/GraniteAttention.

        if hybrid_cache is not None:
            cache_offset = hybrid_cache.length
            k, v = hybrid_cache.update_attention(layer_idx, k, v)
            k, v = k.to(q.dtype), v.to(q.dtype)
        else:
            cache_offset = 0

        # enable_gqa: SDPA shares each k/v head across its query-head group itself - no
        # repeat_interleave copy of the whole cached k/v per step (measured on this project's
        # laptop: at a 200-token context that copy cost ~90 ms per generated token over 28
        # layers, vs ~7 ms without it; the gap grows with the conversation).
        if hybrid_cache is not None:
            mask = _causal_mask(seq_len, k.shape[-2], cache_offset)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.n_head * self.head_dim)
        return self.o_proj(out)


class NemotronHMLP(nn.Module):
    """`down_proj(relu(up_proj(x))**2)` - non-gated, squared-ReLU activation (confirmed real
    `mlp_hidden_act = "relu2"` in `modeling_nemotron_h.py`) - no `ffn_gate` tensor at all in the
    real GGUF file, so this is a new, simpler class, not a reuse of the shared gated `SwiGLUMLP`.
    """

    def __init__(self, n_embd: int, ffn_len: int, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.up_proj = nn.Linear(n_embd, ffn_len, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn_len, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.relu(self.up_proj(x)).pow(2))


class NemotronHMambaBlock(nn.Module):
    """`x + mixer(input_layernorm(x))` - the same single shared pre-norm every real layer type
    uses (confirmed: `attn_norm.weight` is the one real tensor name shared by Mamba/attention/MLP
    layers alike - no separate post-norm anywhere in this architecture)."""

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
    """One real layer's construction, dispatched by its real per-index type (see
    `NemotronHArchitecture.__init__`'s own derivation of `layer_types` from the two real GGUF
    per-layer arrays) - kept here, next to the layer classes themselves, so the architecture file
    stays focused on metadata parsing/materialization orchestration."""
    if layer_type == "mamba":
        mixer = NemotronHMamba2Mixer(
            n_embd, mamba_d_inner, mamba_num_heads, mamba_head_dim, d_state, n_group,
            conv_kernel, rms_eps, dtype=dtype,
        )
        return NemotronHMambaBlock(n_embd, rms_eps, mixer, dtype=dtype)
    if layer_type == "attention":
        attn = NemotronHAttention(n_embd, n_head, n_head_kv, head_dim, dtype=dtype)
        return NemotronHAttentionBlock(n_embd, rms_eps, attn, dtype=dtype)
    mlp = NemotronHMLP(n_embd, ffn_len, dtype=dtype)
    return NemotronHMLPBlock(n_embd, rms_eps, mlp, dtype=dtype)


def materialize_mamba_layer(
    loader: GGUFModelLoader, prefix: str, mixer: NemotronHMamba2Mixer
) -> None:
    """Always a plain `.copy_()` - see `NemotronHArchitecture`'s own docstring for why
    `_load_projection` (the quantized-native-eligible path) never applies to the SSM block."""
    mixer.in_proj.weight.copy_(loader.load_tensor(prefix + "ssm_in.weight"))
    mixer.conv1d_weight.copy_(loader.load_tensor(prefix + "ssm_conv1d.weight"))
    mixer.conv1d_bias.copy_(loader.load_tensor(prefix + "ssm_conv1d.bias"))
    mixer.dt_bias.copy_(loader.load_tensor(prefix + "ssm_dt.bias"))
    mixer.a.copy_(loader.load_tensor(prefix + "ssm_a").reshape(-1))
    mixer.d.copy_(loader.load_tensor(prefix + "ssm_d").reshape(-1))
    mixer.norm_weight.copy_(loader.load_tensor(prefix + "ssm_norm.weight"))
    mixer.out_proj.weight.copy_(loader.load_tensor(prefix + "ssm_out.weight"))


def materialize_attention_layer(
    loader: GGUFModelLoader,
    prefix: str,
    attn: NemotronHAttention,
    load_projection: Callable[..., nn.Module],
    dtype: torch.dtype,
    enabled: bool,
) -> None:
    # No unpermute_rope_rows() call anywhere - that permutation exists solely for rotate-half
    # RoPE, which this architecture never applies.
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
