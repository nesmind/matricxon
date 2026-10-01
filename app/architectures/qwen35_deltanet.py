import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.qwen35_delta_kernels import gated_delta_rule

_L2_EPS = 1e-6


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + _L2_EPS)


class Qwen35GatedDeltaNet(nn.Module):
    """Qwen3.5's linear-attention mixer (3 of every 4 layers) - a causal depthwise conv over a
    fused [q | k | v] projection, then the gated delta rule (`gated_delta_rule`), then a per-head
    RMSNorm gated by `silu(z)`. Confirmed against the real `qwen35` GGUF header (tensor names and
    shapes of `Qwen3.5-9B`, 2026-10-01) and HF's `Qwen3NextGatedDeltaNet`.

    GGUF tensor -> role: `attn_qkv` (fused q/k/v), `attn_gate` (z), `ssm_beta` (write strength,
    through a sigmoid), `ssm_alpha` (decay input), `ssm_dt.bias`, `ssm_a`, `ssm_conv1d` (no bias),
    `ssm_norm` (per value-head-dim weight, used as-is - no `1 +`), `ssm_out`.

    `ssm_a` is used directly as `A` (llama.cpp's converter already bakes in HF's `-exp(A_log)`,
    same as nemotron_h's `ssm_a`): `g = A * softplus(alpha + dt_bias)`.

    Key heads (`n_k_heads`) are fewer than value heads (`n_v_heads`); each is shared across
    `n_v_heads // n_k_heads` value heads. `tiled_heads` picks the expansion order: True tiles
    (value head j uses key head j % n_k_heads, llama.cpp's ggml_repeat layout), False interleaves
    (j // ratio, HF's layout).
    """

    def __init__(
        self,
        n_embd: int,
        n_k_heads: int,
        n_v_heads: int,
        head_dim: int,
        conv_kernel: int,
        rms_eps: float,
        tiled_heads: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_k_heads = n_k_heads
        self.n_v_heads = n_v_heads
        self.head_dim = head_dim
        self.conv_kernel = conv_kernel
        self.rms_eps = rms_eps
        self.tiled_heads = tiled_heads
        self.key_dim = n_k_heads * head_dim
        self.value_dim = n_v_heads * head_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.qkv_proj = nn.Linear(n_embd, self.conv_dim, bias=False, dtype=dtype)
        self.z_proj = nn.Linear(n_embd, self.value_dim, bias=False, dtype=dtype)
        self.beta_proj = nn.Linear(n_embd, n_v_heads, bias=False, dtype=dtype)
        self.alpha_proj = nn.Linear(n_embd, n_v_heads, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.value_dim, n_embd, bias=False, dtype=dtype)
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, conv_kernel, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(n_v_heads, dtype=dtype))
        self.a = nn.Parameter(torch.empty(n_v_heads, dtype=dtype))
        self.norm_weight = nn.Parameter(torch.empty(head_dim, dtype=dtype))

    def _expand_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(T, n_k_heads, d) -> (T, n_v_heads, d)."""
        ratio = self.n_v_heads // self.n_k_heads
        if ratio == 1:
            return x
        if self.tiled_heads:
            return x.repeat(1, ratio, 1)
        return x.repeat_interleave(ratio, dim=1)

    def _causal_conv(
        self, mixed: torch.Tensor, conv_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Depthwise causal conv + silu over (1, T, conv_dim), carrying the last k-1 inputs."""
        padded = torch.cat([conv_state.transpose(1, 2), mixed.transpose(1, 2)], dim=-1)
        out = F.conv1d(padded, self.conv1d_weight.unsqueeze(1), groups=self.conv_dim)
        new_state = padded[:, :, -(self.conv_kernel - 1) :].transpose(1, 2)
        return F.silu(out).transpose(1, 2), new_state

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        _, seq_len, _ = x.shape  # batch == 1, project-wide invariant
        conv_state, ssm_state = hybrid_cache.mamba_state(layer_idx)
        mixed, new_conv_state = self._causal_conv(self.qkv_proj(x), conv_state)

        q, k, v = mixed.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(seq_len, self.n_k_heads, self.head_dim).float()
        k = k.reshape(seq_len, self.n_k_heads, self.head_dim).float()
        v = v.reshape(seq_len, self.n_v_heads, self.head_dim).float()
        q = self._expand_heads(_l2norm(q)) * self.head_dim**-0.5
        k = self._expand_heads(_l2norm(k))

        beta = torch.sigmoid(self.beta_proj(x)[0].float())
        g = self.a.float() * F.softplus(self.alpha_proj(x)[0].float() + self.dt_bias.float())
        out, state = gated_delta_rule(q, k, v, g, beta, ssm_state[0].float().clone())

        # Per-head RMSNorm (weight as-is), then gate by silu(z).
        out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + self.rms_eps)
        out = out * self.norm_weight.float()
        z = self.z_proj(x)[0].float().view(seq_len, self.n_v_heads, self.head_dim)
        out = (out * F.silu(z)).reshape(1, seq_len, self.value_dim).to(x.dtype)

        hybrid_cache.set_mamba_state(
            layer_idx, new_conv_state, state.unsqueeze(0).to(ssm_state.dtype)
        )
        return self.out_proj(out)
