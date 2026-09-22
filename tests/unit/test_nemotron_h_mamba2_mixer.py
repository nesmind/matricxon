"""Standalone correctness tests for `NemotronHMamba2Mixer`
(`app/architectures/nemotron_h_layers.py`) - isolates the real selective-state-space recurrence,
causal conv1d, discretization, and gate-before-norm ordering from GGUF loading and the rest of the
architecture, per the real risk this module's own docstring calls out (same treatment
`tests/unit/test_granitemoe_ffn.py` gave GraniteMoE's router logic).

`_reference_forward` below is a deliberately independent reimplementation - manual per-timestep
causal convolution (elementwise multiply-sum, not `F.conv1d`) and manual per-timestep state-update
broadcasting (not `torch.einsum`) - re-derived from this project's own verified real-Mamba-2-math
notes rather than calling into `NemotronHMamba2Mixer`'s own code at all.
"""

import torch
import torch.nn.functional as F

from app.architectures.nemotron_h_mamba2 import NemotronHMamba2Mixer
from app.runtime.mamba_cache import NemotronHHybridCache

N_EMBD = 6
N_HEADS = 4
HEAD_DIM = 3
D_INNER = N_HEADS * HEAD_DIM  # 12
D_STATE = 4
N_GROUPS = 2
CONV_KERNEL = 3
NORM_EPS = 1e-5


def _random_mixer(seed: int = 0) -> NemotronHMamba2Mixer:
    torch.manual_seed(seed)
    mixer = NemotronHMamba2Mixer(
        N_EMBD, D_INNER, N_HEADS, HEAD_DIM, D_STATE, N_GROUPS, CONV_KERNEL, NORM_EPS,
        dtype=torch.float32,
    )
    with torch.no_grad():
        mixer.in_proj.weight.copy_(torch.randn_like(mixer.in_proj.weight) * 0.1)
        mixer.conv1d_weight.copy_(torch.randn_like(mixer.conv1d_weight) * 0.1)
        mixer.conv1d_bias.copy_(torch.randn_like(mixer.conv1d_bias) * 0.1)
        mixer.dt_bias.copy_(torch.randn_like(mixer.dt_bias) * 0.1)
        # Real ssm_a values are always negative (confirmed against a real downloaded GGUF file,
        # see the module's own docstring) - used directly here too, no -exp() transform.
        mixer.a.copy_(-torch.rand_like(mixer.a) * 2.0 - 0.01)
        mixer.d.copy_(torch.randn_like(mixer.d) * 0.1)
        mixer.norm_weight.copy_(
            torch.ones_like(mixer.norm_weight) + torch.randn_like(mixer.norm_weight) * 0.05
        )
        mixer.out_proj.weight.copy_(torch.randn_like(mixer.out_proj.weight) * 0.1)
    return mixer


def _fresh_cache() -> NemotronHHybridCache:
    return NemotronHHybridCache(
        layer_types=["mamba"],
        attention_layer_shape=(1, 1),
        mamba_conv_state_shape=(CONV_KERNEL - 1, D_INNER + 2 * N_GROUPS * D_STATE),
        mamba_ssm_state_shape=(N_HEADS, HEAD_DIM, D_STATE),
        max_seq_len=32,
    )


def _reference_scan(
    x: torch.Tensor, mixer: NemotronHMamba2Mixer, conv_state: torch.Tensor, ssm_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (raw_scan_output y, gate, new_conv_state, new_ssm_state) - everything needed by
    both the gate-then-norm (correct) and norm-then-gate (wrong) orderings below, computed once."""
    seq_len = x.shape[1]
    conv_dim = mixer.conv_dim
    proj = x[0] @ mixer.in_proj.weight.T  # (seq_len, in_proj_out)
    gate = proj[:, : mixer.d_inner]
    hidden_bc = proj[:, mixer.d_inner : mixer.d_inner + conv_dim]
    dt_raw = proj[:, mixer.d_inner + conv_dim :]

    history = conv_state[0].clone()  # (conv_kernel - 1, conv_dim)
    conv_outputs = torch.empty(seq_len, conv_dim)
    for t in range(seq_len):
        window = torch.cat([history, hidden_bc[t : t + 1]], dim=0)  # (conv_kernel, conv_dim)
        conv_outputs[t] = (window.T * mixer.conv1d_weight).sum(dim=-1) + mixer.conv1d_bias
        history = window[1:]
    new_conv_state = history.unsqueeze(0)

    hidden_bc_conv = F.silu(conv_outputs)  # (seq_len, conv_dim)
    x_ssm = hidden_bc_conv[:, : mixer.d_inner].view(seq_len, N_HEADS, HEAD_DIM)
    b = hidden_bc_conv[:, mixer.d_inner : mixer.d_inner + N_GROUPS * D_STATE].view(
        seq_len, N_GROUPS, D_STATE
    )
    c = hidden_bc_conv[:, mixer.d_inner + N_GROUPS * D_STATE :].view(seq_len, N_GROUPS, D_STATE)
    heads_per_group = N_HEADS // N_GROUPS
    b = b.repeat_interleave(heads_per_group, dim=1)
    c = c.repeat_interleave(heads_per_group, dim=1)

    dt = F.softplus(dt_raw + mixer.dt_bias)  # (seq_len, n_heads)
    a = mixer.a  # used directly, no transform

    state = ssm_state[0].clone()
    outputs = torch.empty(seq_len, N_HEADS, HEAD_DIM)
    for t in range(seq_len):
        decay = torch.exp(dt[t] * a).view(-1, 1, 1)
        outer = dt[t].view(-1, 1, 1) * (b[t].unsqueeze(1) * x_ssm[t].unsqueeze(2))
        state = state * decay + outer
        outputs[t] = (state * c[t].unsqueeze(1)).sum(dim=-1) + mixer.d.view(-1, 1) * x_ssm[t]

    y = outputs.reshape(seq_len, mixer.d_inner)
    return y, gate, new_conv_state, state.unsqueeze(0)


def _apply_gate_then_norm(
    y: torch.Tensor, gate: torch.Tensor, mixer: NemotronHMamba2Mixer
) -> torch.Tensor:
    y_gated = y * F.silu(gate)
    return _grouped_rms_norm(y_gated, mixer)


def _apply_norm_then_gate(
    y: torch.Tensor, gate: torch.Tensor, mixer: NemotronHMamba2Mixer
) -> torch.Tensor:
    y_normed = _grouped_rms_norm(y, mixer)
    return y_normed * F.silu(gate)


def _grouped_rms_norm(y: torch.Tensor, mixer: NemotronHMamba2Mixer) -> torch.Tensor:
    seq_len = y.shape[0]
    y = y.view(seq_len, N_GROUPS, mixer.d_inner // N_GROUPS)
    variance = y.pow(2).mean(dim=-1, keepdim=True)
    y = y * torch.rsqrt(variance + NORM_EPS)
    return (y * mixer.norm_weight).reshape(seq_len, mixer.d_inner)


def _reference_forward(
    x: torch.Tensor, mixer: NemotronHMamba2Mixer, conv_state: torch.Tensor, ssm_state: torch.Tensor
) -> torch.Tensor:
    y, gate, _, _ = _reference_scan(x, mixer, conv_state, ssm_state)
    y_normed = _apply_gate_then_norm(y, gate, mixer)
    return (y_normed @ mixer.out_proj.weight.T).unsqueeze(0)


class TestMatchesReferenceRecurrence:
    def test_multi_token_prefill(self) -> None:
        mixer = _random_mixer()
        cache = _fresh_cache()
        x = torch.randn(1, 5, N_EMBD) * 0.1

        actual = mixer(x, cache, layer_idx=0)
        expected = _reference_forward(x, mixer, *_fresh_cache().mamba_state(0))

        assert torch.allclose(actual, expected, atol=1e-4)

    def test_single_decode_token_with_nonzero_carried_state(self) -> None:
        """The dominant real workload shape - a decode step continuing from real, non-zero
        conv/ssm state carried over from an earlier prefill, not a fresh zero-initialized cache."""
        mixer = _random_mixer()
        real_cache = _fresh_cache()
        ref_conv, ref_ssm = _fresh_cache().mamba_state(0)

        prefill = torch.randn(1, 4, N_EMBD) * 0.1
        mixer(prefill, real_cache, layer_idx=0)
        _, _, ref_conv, ref_ssm = _reference_scan(prefill, mixer, ref_conv, ref_ssm)

        decode_token = torch.randn(1, 1, N_EMBD) * 0.1
        actual = mixer(decode_token, real_cache, layer_idx=0)
        expected = _reference_forward(decode_token, mixer, ref_conv, ref_ssm)

        assert torch.allclose(actual, expected, atol=1e-4)


class TestGatingHappensBeforeNorm:
    def test_real_module_matches_gate_then_norm_not_norm_then_gate(self) -> None:
        mixer = _random_mixer()
        x = torch.randn(1, 4, N_EMBD) * 0.1
        conv_state, ssm_state = _fresh_cache().mamba_state(0)

        y, gate, _, _ = _reference_scan(x, mixer, conv_state, ssm_state)
        gate_then_norm = _apply_gate_then_norm(y, gate, mixer) @ mixer.out_proj.weight.T
        norm_then_gate = _apply_norm_then_gate(y, gate, mixer) @ mixer.out_proj.weight.T
        gate_then_norm = gate_then_norm.unsqueeze(0)
        norm_then_gate = norm_then_gate.unsqueeze(0)

        actual = mixer(x, _fresh_cache(), layer_idx=0)

        assert torch.allclose(actual, gate_then_norm, atol=1e-4)
        assert not torch.allclose(actual, norm_then_gate, atol=1e-3)


class TestDiscretizationUsesSoftplus:
    def test_dt_activation_is_softplus_not_a_different_function(self) -> None:
        mixer = _random_mixer()
        x = torch.randn(1, 3, N_EMBD) * 0.1
        conv_state, ssm_state = _fresh_cache().mamba_state(0)

        expected = _reference_forward(x, mixer, conv_state, ssm_state)
        actual = mixer(x, _fresh_cache(), layer_idx=0)
        assert torch.allclose(actual, expected, atol=1e-4)

        # A reference using ReLU instead of softplus for dt must NOT match the real module -
        # pins the specific activation, not just "some monotonic nonlinearity."
        y, gate, _, _ = _reference_scan(x, mixer, conv_state, ssm_state)
        # Recompute the scan with relu(dt) instead of softplus(dt) by patching dt_bias math
        # inline: cheapest way to prove the distinction without duplicating the whole scan is to
        # confirm relu and softplus disagree on this mixer's own real dt_raw range, then trust
        # the already-passing exact-match assertion above as the real discriminator.
        proj = x[0] @ mixer.in_proj.weight.T
        dt_raw = proj[:, mixer.d_inner + mixer.conv_dim :]
        softplus_dt = F.softplus(dt_raw + mixer.dt_bias)
        relu_dt = F.relu(dt_raw + mixer.dt_bias)
        assert not torch.allclose(softplus_dt, relu_dt, atol=1e-3)
