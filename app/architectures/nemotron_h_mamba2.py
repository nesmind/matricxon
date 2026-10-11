import torch
import torch.nn.functional as F
from torch import nn


class NemotronHMamba2Mixer(nn.Module):
    """Mamba-2 selective-state-space mixer (HF `NemotronHMamba2Mixer`,; stock Mamba-2
    math, no Nemotron-specific deltas).

    `in_proj(x)` is one fused projection split into `gate` (`d_inner`, multiplies the scan output
    before normalization), `hidden_states_B_C` (`d_inner + 2*n_groups*d_state`, the x/B/C inputs to
    the causal conv1d) and `dt_raw` (`n_heads`, the per-head step before `softplus`).

    `A` (GGUF `ssm_a`, `(n_heads)`) is used directly, with no `-exp`: a real `blk.0.ssm_a` was
    already negative (e.g. `-345.07`, `-0.00038`), not HF's `a_log`; llama.cpp's
    converter bakes the transform in. Applying `-exp` again gives a wrong decay rate, hence the
    name `a`, not `a_log`.

    The recurrence is a sequential loop (same result as the chunked "SSD" parallel scan): `state =
    state * exp(dt*A) + dt * outer(B, x)`, `y = state @ C + D*x`. It serves prefill and one-token
    decode alike (calls are always `batch == 1`), following this project's "correctness first"
    precedent (`QuantizedLinear`'s GEMV kernels, `GraniteMoeFFN`'s sparse dispatch).

    Gating happens before normalization (HF order): `y_gated = y * silu(gate)`, then a grouped
    RMSNorm over `n_groups` groups with `ssm_norm.weight`; not normalize-then-gate.
    """

    def __init__(
        self,
        n_embd: int,
        d_inner: int,
        n_heads: int,
        head_dim: int,
        d_state: int,
        n_groups: int,
        conv_kernel: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_state = d_state
        self.n_groups = n_groups
        self.conv_kernel = conv_kernel
        self.norm_eps = rms_eps
        self.conv_dim = d_inner + 2 * n_groups * d_state

        in_proj_out = 2 * d_inner + 2 * n_groups * d_state + n_heads
        self.in_proj = nn.Linear(n_embd, in_proj_out, bias=False, dtype=dtype)
        # Raw params, not nn.Conv1d: loaded from blk.N.ssm_conv1d.weight/.bias (shape (conv_dim,
        # conv_kernel) after GGUFModelLoader's reversed(ne[]) convention).
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, conv_kernel, dtype=dtype))
        self.conv1d_bias = nn.Parameter(torch.empty(self.conv_dim, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        self.a = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        self.d = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        # (n_groups, d_inner // n_groups): matches ssm_norm.weight's reversed shape.
        self.norm_weight = nn.Parameter(torch.empty(n_groups, d_inner // n_groups, dtype=dtype))
        self.out_proj = nn.Linear(d_inner, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        _, seq_len, _ = x.shape
        proj = self.in_proj(x)
        gate, hidden_bc, dt_raw = proj.split([self.d_inner, self.conv_dim, self.n_heads], dim=-1)

        conv_state, ssm_state = hybrid_cache.mamba_state(layer_idx)
        hidden_bc_t = hidden_bc.transpose(1, 2)
        padded = torch.cat([conv_state.transpose(1, 2), hidden_bc_t], dim=-1)
        conv_out = F.conv1d(
            padded, self.conv1d_weight.unsqueeze(1), bias=self.conv1d_bias, groups=self.conv_dim
        )
        new_conv_state = padded[:, :, -(self.conv_kernel - 1) :].transpose(1, 2)
        hidden_bc = F.silu(conv_out).transpose(1, 2)

        x_ssm, b, c = hidden_bc.split(
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1
        )
        dt = F.softplus(dt_raw + self.dt_bias)
        a = self.a.float()  # final negative A, no transform (see docstring)

        x_ssm = x_ssm.view(1, seq_len, self.n_heads, self.head_dim)
        heads_per_group = self.n_heads // self.n_groups
        b = b.view(1, seq_len, self.n_groups, self.d_state).repeat_interleave(
            heads_per_group, dim=2
        )
        c = c.view(1, seq_len, self.n_groups, self.d_state).repeat_interleave(
            heads_per_group, dim=2
        )

        state = ssm_state[0].to(torch.float32).clone()
        outputs = torch.empty(
            seq_len, self.n_heads, self.head_dim, dtype=torch.float32, device=x.device
        )
        for t in range(seq_len):
            dt_t = dt[0, t].float()
            decay = torch.exp(dt_t * a).view(-1, 1, 1)
            b_t = b[0, t].float()
            x_t = x_ssm[0, t].float()
            c_t = c[0, t].float()
            state = state * decay + dt_t.view(-1, 1, 1) * torch.einsum("hs,hp->hps", b_t, x_t)
            outputs[t] = torch.einsum("hps,hs->hp", state, c_t) + self.d.float().view(-1, 1) * x_t

        y = outputs.to(x.dtype).reshape(1, seq_len, self.d_inner)
        y_gated = y * F.silu(gate)  # gate BEFORE norm (see docstring)
        y_normed = self._grouped_rms_norm(y_gated)

        hybrid_cache.set_mamba_state(
            layer_idx, new_conv_state, state.unsqueeze(0).to(ssm_state.dtype)
        )
        return self.out_proj(y_normed)

    def _grouped_rms_norm(self, y: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = y.shape
        y_dtype = y.dtype
        y = y.view(batch, seq_len, self.n_groups, self.d_inner // self.n_groups).to(torch.float32)
        variance = y.pow(2).mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(variance + self.norm_eps)
        return (y.to(y_dtype) * self.norm_weight).reshape(batch, seq_len, self.d_inner)
