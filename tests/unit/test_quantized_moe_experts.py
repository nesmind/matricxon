"""QuantizedMoEExperts (app/architectures/moe_experts.py) - the shared, quantized-native-aware
MoE expert FFN every real MoE architecture in this project now composes. Exercised through
GraniteMoeArchitecture (the simplest real MoE architecture - no shared dense branch, no PLE/
cross-layer-KV complexity) on a real, tiny Q8_0-quantized GGUF - the same "packed vs dense, same
weights, same output" precedent tests/unit/test_packed_weights.py already established for the
plain 2D case, now proven for a real 3D per-expert tensor too.
"""

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
    # The big dense placeholder is freed once every expert packs - see _materialize_one's own
    # docstring for why this differs from the plain 2D placeholder-then-replace precedent.
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
    """The router is small and always a plain `.copy_()` - see QuantizedMoEExperts's own
    docstring for why packing it wouldn't meaningfully help."""
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
    """Every real per-expert QuantizedLinear must be a discoverable submodule (see
    QuantizedMoEExperts's own docstring) - otherwise ModelArchitecture.close()'s own
    _release_packed_modules walk would miss it, and mmap.close() would raise a real
    BufferError instead of this test just passing."""
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    model.close()  # must not raise


def test_expert_per_tok_selection_still_works_when_packed(tmp_path):
    """Real sparse dispatch (only EXPERTS_PER_TOK of N_EXPERTS ever computed per token) - proves
    the packed path is exercised for a genuinely selective, not-every-expert case too."""
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
    """The module-level `numba_backend` fixture forces every other test in this file onto the
    Numba path - this one explicitly re-activates a real native C backend afterward (Q8_0 has
    had native coverage since before this session, see ROADMAP.md) to prove
    `QuantizedMoEExperts` gets that backend "for free" too: it reuses `QuantizedLinear` per
    expert unchanged, and that class's own native-vs-Numba-vs-dequant dispatch already doesn't
    care whether the tensor it's given is a whole dense weight or one expert's own slice of a
    bigger 3D one."""
    monkeypatch.setattr(NativeGemm, "_active", native_gemm)
    path = build_tiny_granitemoe_q8_gguf(tmp_path / "tiny-granitemoe-q8.gguf")
    dense = _load(path, packed=False)
    packed = _load(path, packed=True)
    with torch.no_grad():
        expected = dense.forward(INPUT_IDS)
        result = packed.forward(INPUT_IDS)

    # Looser than test_native_gemm.py's own 1e-2 bound for a single quantized matmul - here the
    # activations pass through 3 chained native-C quantized matmuls per expert (gate, up, down),
    # each independently contributing its own int8-activation-quantization rounding, so the
    # compounded error is expected to be larger. Confirmed stable at ~1.5-2% across 5 random
    # seeds (not a data-dependent slicing bug, which would show wildly varying/much larger error).
    rel_err = torch.linalg.norm(result - expected) / torch.linalg.norm(expected)
    assert rel_err < 3e-2
    for expert_id in range(N_EXPERTS):
        assert packed.layers[0].mlp.experts.gate_packed[str(expert_id)]._native is native_gemm
    packed.close()
    dense.close()
