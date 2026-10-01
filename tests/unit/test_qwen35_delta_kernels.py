"""Numba and C Gated DeltaNet kernels must match the plain torch reference (and each other)."""

import pytest
import torch

from app.architectures.qwen35_delta_kernels import (
    gated_delta_rule,
    gated_delta_rule_reference,
)
from app.native.gemm import NativeGemm

T, H, DK, DV = 9, 4, 16, 8


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return (
        torch.randn(T, H, DK) * 0.3,
        torch.randn(T, H, DK) * 0.3,
        torch.randn(T, H, DV),
        -torch.rand(T, H) * 2,
        torch.rand(T, H),
        torch.randn(H, DK, DV) * 0.1,
    )


def _check(backend: str) -> None:
    NativeGemm.configure(backend)
    if backend == "native" and NativeGemm.active() is None:
        pytest.skip("native kernels unavailable")
    q, k, v, g, beta, state = _inputs()
    ref_out, ref_state = gated_delta_rule_reference(q, k, v, g, beta, state.clone())
    out, new_state = gated_delta_rule(q, k, v, g, beta, state.clone())
    assert torch.allclose(out, ref_out, atol=1e-5)
    assert torch.allclose(new_state, ref_state, atol=1e-5)


@pytest.mark.parametrize("backend", ["numba", "native"])
def test_kernel_matches_torch_reference(backend: str) -> None:
    try:
        _check(backend)
    finally:
        NativeGemm.configure("numba")
