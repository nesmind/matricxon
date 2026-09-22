import torch
import torch.nn.functional as F
from torch import nn


class NemotronHMamba2Mixer(nn.Module):
    """Real Mamba-2 selective-state-space mixer (confirmed against HF `transformers`' real
    `modeling_nemotron_h.py`/`NemotronHMamba2Mixer`, 2026-09-22 - stock Mamba-2 math, no real
    Nemotron-specific deltas found in the SSM layer itself).

    `in_proj(x)` splits into three pieces in one fused projection: `gate` (size `d_inner`,
    multiplies the scan output *before* normalization - see the real gating order below),
    `hidden_states_B_C` (size `d_inner + 2*n_groups*d_state` - the x/B/C inputs to the causal
    conv1d, concatenated), and `dt_raw` (size `n_heads`, the per-head discretization step before
    its real `softplus` activation).

    `A` (real GGUF tensor `ssm_a`, shape `(n_heads,)`) is used **directly, with no `-exp()`
    transform** - confirmed by range-reading the real bytes of a real
    `nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF` file's `blk.0.ssm_a` tensor (2026-09-22): every real
    value was already negative (e.g. `-345.07`, `-0.00038`, `-5569.16`), not the small positive
    `a_log` values HF's own Python reference computes `A = -exp(a_log)` from at *module* level -
    llama.cpp's converter bakes that transform in once at conversion time instead of at every real
    forward pass. Applying `-exp()` again here would silently produce a wrong (and, given the real
    magnitude range observed, wildly wrong) decay rate - this is the one real, easy-to-get-
    backwards correctness trap this module's own construction guards against by naming the
    parameter plain `a`, not `a_log`.

    The real recurrence - mathematically identical whether computed via the real chunked
    "SSD" parallel-scan algorithm (a performance optimization, not a different formula) or a
    plain sequential per-timestep loop - is implemented here as a sequential loop: `state = state
    * exp(dt*A) + dt * outer(B, x)`, `y = state @ C + D*x`, correct for both a many-token prefill
    and a one-token decode step via the identical code path (matricxon's own confirmed project-
    wide call shape is always `batch == 1`, single sequence - see `QuantizedLinear`'s own
    docstring for the same invariant elsewhere). This project's own established "correctness
    first, honest real performance profile over a from-scratch fused kernel" precedent
    (`QuantizedLinear`'s real GEMV kernels, `GraniteMoeFFN`'s sparse-not-fused expert dispatch)
    is why a plain loop, not the real chunked SSD algorithm, is the right first cut here too.

    Gating happens **before** normalization (confirmed exact order from the real HF source):
    `y_gated = y * silu(gate)`, then a grouped RMSNorm over `n_groups` groups of the `d_inner`
    channels using the real `ssm_norm.weight` tensor - NOT normalize-then-gate.
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
        # Raw params, not nn.Conv1d - loaded straight from the real blk.N.ssm_conv1d.weight/.bias
        # tensors (real shape confirmed (conv_dim, conv_kernel) after GGUFModelLoader's own
        # reversed(ne[]) convention).
        self.conv1d_weight = nn.Parameter(torch.empty(self.conv_dim, conv_kernel, dtype=dtype))
        self.conv1d_bias = nn.Parameter(torch.empty(self.conv_dim, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        self.a = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        self.d = nn.Parameter(torch.empty(n_heads, dtype=dtype))
        # (n_groups, d_inner // n_groups) - matches ssm_norm.weight's real reversed shape.
        self.norm_weight = nn.Parameter(torch.empty(n_groups, d_inner // n_groups, dtype=dtype))
        self.out_proj = nn.Linear(d_inner, n_embd, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, hybrid_cache: object, layer_idx: int) -> torch.Tensor:
        _, seq_len, _ = x.shape  # batch == 1, project-wide invariant
        proj = self.in_proj(x)
        gate, hidden_bc, dt_raw = proj.split(
            [self.d_inner, self.conv_dim, self.n_heads], dim=-1
        )

        conv_state, ssm_state = hybrid_cache.mamba_state(layer_idx)
        hidden_bc_t = hidden_bc.transpose(1, 2)  # (1, conv_dim, seq_len)
        padded = torch.cat([conv_state.transpose(1, 2), hidden_bc_t], dim=-1)
        conv_out = F.conv1d(
            padded, self.conv1d_weight.unsqueeze(1), bias=self.conv1d_bias, groups=self.conv_dim
        )
        new_conv_state = padded[:, :, -(self.conv_kernel - 1) :].transpose(1, 2)
        hidden_bc = F.silu(conv_out).transpose(1, 2)  # (1, seq_len, conv_dim)

        x_ssm, b, c = hidden_bc.split(
            [self.d_inner, self.n_groups * self.d_state, self.n_groups * self.d_state], dim=-1
        )
        dt = F.softplus(dt_raw + self.dt_bias)  # (1, seq_len, n_heads)
        a = self.a.float()  # already the real, final, negative A - no transform (see docstring)

        x_ssm = x_ssm.view(1, seq_len, self.n_heads, self.head_dim)
        heads_per_group = self.n_heads // self.n_groups
        b = b.view(1, seq_len, self.n_groups, self.d_state).repeat_interleave(
            heads_per_group, dim=2
        )
        c = c.view(1, seq_len, self.n_groups, self.d_state).repeat_interleave(
            heads_per_group, dim=2
        )

        state = ssm_state[0].to(torch.float32).clone()  # (n_heads, head_dim, d_state)
        outputs = torch.empty(seq_len, self.n_heads, self.head_dim, dtype=torch.float32)
        for t in range(seq_len):
            dt_t = dt[0, t].float()  # (n_heads,)
            decay = torch.exp(dt_t * a).view(-1, 1, 1)
            b_t = b[0, t].float()  # (n_heads, d_state)
            x_t = x_ssm[0, t].float()  # (n_heads, head_dim)
            c_t = c[0, t].float()  # (n_heads, d_state)
            state = state * decay + dt_t.view(-1, 1, 1) * torch.einsum("hs,hp->hps", b_t, x_t)
            outputs[t] = torch.einsum("hps,hs->hp", state, c_t) + self.d.float().view(-1, 1) * x_t

        y = outputs.to(x.dtype).reshape(1, seq_len, self.d_inner)
        y_gated = y * F.silu(gate)  # gate BEFORE norm - see docstring
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
