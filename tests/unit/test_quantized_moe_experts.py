"""QuantizedMoEExperts (app/architectures/moe_experts.py), the shared quantized-aware MoE expert
FFN. Exercised through GraniteMoeArchitecture on a tiny Q8_0 GGUF: packed vs dense must give the
same output, now for 3D per-expert tensors."""

import pytest
import torch

from app.architectures.granitemoe import GraniteMoeArchitecture
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.models.load_dtype import estimate_quantized_native_bytes
from app.native.gemm import NativeGemm
from app.native.library import NativeKernelLibrary
from tests.tiny_gguf_granitemoe_q8 import (
    EXPERTS_PER_TOK,
    N_EXPERTS,
    build_tiny_granitemoe_q8_gguf,
)

INPUT_IDS = torch.tensor([[1, 2, 3, 4]])


@pytest.fixture(autouse=True)
def numba_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NativeGemm, "_active", None)


def _load(path, packed: bool) -> GraniteMoeArchitecture:
    loader = GGUFModelLoader(path, dtype=torch.float32)
    return GraniteMoeArchitecture.from_gguf(
        loader, dtype=torch.float32, enable_quantized_native=packed
    )


def test_packed_model_matches_the_dense_model(tmp_path):
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    dense = _load(path, packed=False)
    packed = _load(path, packed=True)
    with torch.no_grad():
        expected = dense.forward(INPUT_IDS)
        result = packed.forward(INPUT_IDS)

    assert torch.allclose(result, expected, atol=1e-4)
    packed.close()
    dense.close()


def test_packed_model_builds_a_quantized_linear_per_expert(tmp_path):
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    experts = model.layers[0].mlp.experts
    assert len(experts.gate_packed) == N_EXPERTS
    assert len(experts.up_packed) == N_EXPERTS
    assert len(experts.down_packed) == N_EXPERTS
    for expert_id in range(N_EXPERTS):
        assert isinstance(experts.gate_packed[str(expert_id)], QuantizedLinear)
    # The big dense placeholder is freed once every expert packs (see _materialize_one).
    assert experts.gate_exps.numel() == 0
    assert experts.up_exps.numel() == 0
    assert experts.down_exps.numel() == 0
    model.close()


def test_dense_model_never_builds_any_packed_linear(tmp_path):
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=False)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    experts = model.layers[0].mlp.experts
    assert len(experts.gate_packed) == 0
    assert experts.gate_exps.numel() > 0
    model.close()


def test_router_stays_a_plain_dense_tensor_either_way(tmp_path):
    """The router is small and always a plain `.copy_` (see QuantizedMoEExperts)."""
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    assert model.layers[0].mlp.router.weight.numel() > 0
    model.close()


def test_ram_estimate_counts_packed_expert_tensors_for_granitemoe(tmp_path):
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    tensor_infos = GGUFReader(path).read().tensor_infos
    packed_names = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
    registry = QuantStrategyRegistry()
    saved = sum(
        t.n_elements * 2 - registry.get(t.ggml_type).byte_length(t.n_elements)
        for t in tensor_infos
        if t.name.endswith(packed_names)
    )
    without = estimate_quantized_native_bytes(tensor_infos, enabled=True, architecture_name="")
    with_granitemoe = estimate_quantized_native_bytes(tensor_infos, True, "granitemoe")

    assert saved > 0
    assert with_granitemoe <= without - saved


def test_release_lets_the_mmap_close_cleanly(tmp_path):
    """Every per-expert QuantizedLinear must be a discoverable submodule, else
    `_release_packed_modules` misses it and mmap.close raises BufferError."""
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    model.close()


def test_expert_per_tok_selection_still_works_when_packed(tmp_path):
    """Sparse dispatch (only EXPERTS_PER_TOK of N_EXPERTS computed per token) works on the packed
    path."""
    assert EXPERTS_PER_TOK < N_EXPERTS
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        logits = model.forward(INPUT_IDS)

    assert torch.isfinite(logits).all()
    model.close()


@pytest.fixture(scope="module")
def native_gemm(tmp_path_factory: pytest.TempPathFactory) -> NativeGemm:
    lib = NativeKernelLibrary(build_dir=tmp_path_factory.mktemp("native_build")).load()
    return NativeGemm(lib, n_threads=2)


def test_packed_model_matches_the_dense_model_via_the_native_c_backend(
    tmp_path, monkeypatch: pytest.MonkeyPatch, native_gemm: NativeGemm
) -> None:
    """The module's `numba_backend` fixture forces Numba; this test re-activates the native C
    backend to prove QuantizedMoEExperts gets it too, since it reuses `QuantizedLinear` per
    expert."""
    monkeypatch.setattr(NativeGemm, "_active", native_gemm)
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    dense = _load(path, packed=False)
    packed = _load(path, packed=True)
    with torch.no_grad():
        expected = dense.forward(INPUT_IDS)
        result = packed.forward(INPUT_IDS)

    # Looser than test_native_gemm.py's 1e-2: three chained native matmuls per expert (gate, up,
    # down) compound int8 activation rounding. Stable at ~1.5-2% across 5 seeds.
    rel_err = torch.linalg.norm(result - expected) / torch.linalg.norm(expected)
    assert rel_err < 3e-2
    for expert_id in range(N_EXPERTS):
        assert packed.layers[0].mlp.experts.gate_packed[str(expert_id)]._native is native_gemm
    packed.close()
    dense.close()
