"""Shared-input projections (q/k/v, gate/up) as one native call (`mx_gemm_multi`, `MultiGemmPlan`,
`project_shared`) must give exactly what separate `mx_gemm` calls give, and fall back where they
can't apply. Plus the fused SwiGLU step (`FusedOps.silu_mul`) on both backends."""

import shutil

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.quantized_linear import QuantizedLinear
from app.architectures.shared_projection import project_shared
from app.gguf.constants import GGMLQuantizationType as T
from app.native.fused_ops import FusedOps, _NativeBackend, _NumbaBackend
from app.native.gemm import NativeGemm
from app.native.gemm_multi import MultiGemmPlan
from app.native.library import NativeKernelLibrary
from tests.unit.test_native_gemm import _weight

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")

IN_F = 512


@pytest.fixture(scope="module")
def lib(tmp_path_factory: pytest.TempPathFactory):
    return NativeKernelLibrary(build_dir=tmp_path_factory.mktemp("native_build")).load()


@pytest.fixture
def gemm(lib, monkeypatch: pytest.MonkeyPatch) -> NativeGemm:
    active = NativeGemm(lib, n_threads=2)
    monkeypatch.setattr(NativeGemm, "_active", active)
    return active


def _items(types: list[T], outs: list[int]) -> list[tuple[int, memoryview, int]]:
    return [
        (t, memoryview(_weight(t, out, IN_F, seed=i)), out)
        for i, (t, out) in enumerate(zip(types, outs, strict=True))
    ]


@pytest.mark.parametrize("n_tokens", [1, 4])
@pytest.mark.parametrize(
    "types", [[T.Q4_K, T.Q6_K, T.Q8_0], [T.Q3_K, T.Q5_K], [T.IQ4_NL, T.Q4_K, T.IQ4_XS]]
)
def test_equals_separate_calls_exactly(gemm: NativeGemm, types: list[T], n_tokens: int) -> None:
    items = _items(types, [64, 32, 48][: len(types)])
    x = torch.randn(n_tokens, IN_F, generator=torch.Generator().manual_seed(1))
    together = MultiGemmPlan.build(items, IN_F).run(x)
    for (t, raw, out), y in zip(items, together, strict=True):
        assert torch.equal(y, gemm.matmul(t, raw, x, out, IN_F))


def test_declines_a_prefill_that_needs_unpacked_rows(gemm: NativeGemm) -> None:
    plan = MultiGemmPlan.build(_items([T.Q4_K, T.Q6_K], [32, 32]), IN_F)
    assert plan.run(torch.randn(16, IN_F)) is None


def test_no_plan_without_native_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NativeGemm, "_active", None)
    assert MultiGemmPlan.build(_items([T.Q4_K, T.Q6_K], [32, 32]), IN_F) is None


class _Attention(nn.Module):
    def __init__(self, items: list[tuple[int, memoryview, int]]) -> None:
        super().__init__()
        for name, (t, raw, out) in zip(("q_proj", "k_proj", "v_proj"), items, strict=True):
            setattr(self, name, QuantizedLinear(out, IN_F, t, raw, dtype=torch.float32))


@pytest.mark.parametrize("seq_len", [1, 3, 12])
def test_project_shared_equals_each_module(gemm: NativeGemm, seq_len: int) -> None:
    owner = _Attention(_items([T.Q4_K, T.Q6_K, T.Q4_K], [64, 32, 32]))
    x = torch.randn(1, seq_len, IN_F, generator=torch.Generator().manual_seed(2))
    names = ("q_proj", "k_proj", "v_proj")
    for together, alone in zip(
        project_shared(owner, names, x),
        [owner.q_proj(x), owner.k_proj(x), owner.v_proj(x)],
        strict=True,
    ):  # noqa: E501
        assert together.shape == alone.shape and torch.equal(together, alone)


def test_project_shared_plan_is_cached_and_rebuilt_on_swap(gemm: NativeGemm) -> None:
    owner = _Attention(_items([T.Q4_K, T.Q4_K, T.Q4_K], [32, 32, 32]))
    x = torch.randn(1, 1, IN_F)
    project_shared(owner, ("q_proj", "k_proj", "v_proj"), x)
    first = owner.__dict__["_mx_plan_q_proj_k_proj_v_proj"][1]
    project_shared(owner, ("q_proj", "k_proj", "v_proj"), x)
    assert owner.__dict__["_mx_plan_q_proj_k_proj_v_proj"][1] is first
    owner.v_proj = nn.Linear(IN_F, 32, bias=False)  # a torch weight: no shared call
    q, k, v = project_shared(owner, ("q_proj", "k_proj", "v_proj"), x)
    assert owner.__dict__["_mx_plan_q_proj_k_proj_v_proj"][1] is None
    assert torch.equal(v, owner.v_proj(x)) and torch.equal(q, owner.q_proj(x))


@pytest.mark.parametrize("backend", ["native", "numba"])
def test_silu_mul_matches_torch(lib, backend: str) -> None:
    ops = FusedOps(_NativeBackend(lib, 2) if backend == "native" else _NumbaBackend())
    gate, up = torch.randn(1, 1, 3000) * 4, torch.randn(1, 1, 3000)
    assert torch.allclose(ops.silu_mul(gate, up), F.silu(gate) * up, atol=1e-5, rtol=1e-5)
    assert ops.silu_mul(gate.double(), up.double()) is None
