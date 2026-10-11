"""Gemma4's Mixture-of-Experts block (`app/architectures/gemma4_moe.py`), distinct from
GraniteMoE/Mixtral's (differences: that module's docstring: router pre-norm/pre-scale, per-expert
scale, gelu-tanh experts, shared-dense-plus-sparse-expert addition). No real gemma-4-*-A*B-it GGUF
exists to validate against; proven as PLE/cross-layer-KV-reuse were: structural wiring,
a forward pass, and perturbing the one weight each mechanism reads to confirm the output depends on
it (not silently skipped)."""

import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf_gemma4 import (
    MOE_N_LAYER,
    build_tiny_gemma4_gguf,
    build_tiny_gemma4_moe_gguf,
)

INPUT_IDS = torch.tensor([[1, 5, 9]])


def _load(path):
    loader = GGUFModelLoader(path, dtype=torch.float32)
    return Gemma4Architecture.from_gguf(loader, dtype=torch.float32)


def test_every_layer_gets_a_real_moe_block(tmp_path):
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)

    assert model.is_moe
    for layer in model.layers:
        assert layer.moe is not None


def test_forward_runs_and_produces_finite_correctly_shaped_logits(tmp_path):
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)
    with torch.no_grad():
        logits = model.forward(INPUT_IDS)

    assert logits.shape == (1, INPUT_IDS.shape[1], model.vocab_size)
    assert torch.isfinite(logits).all()
    model.close()


def test_perturbing_an_experts_own_weight_changes_the_output(tmp_path):
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)
    with torch.no_grad():
        baseline_logits = model.forward(INPUT_IDS)
        for layer in model.layers:
            layer.moe.experts.gate_exps.add_(1.0)
        perturbed_logits = model.forward(INPUT_IDS)
    model.close()

    assert not torch.allclose(baseline_logits, perturbed_logits)


def test_perturbing_the_per_expert_scale_changes_the_output(tmp_path):
    """The one parameter GraniteMoE's router lacks: proves it is read, not a dead field."""
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)
    with torch.no_grad():
        baseline_logits = model.forward(INPUT_IDS)
        for layer in model.layers:
            layer.moe.router.per_expert_scale.add_(1.0)
        perturbed_logits = model.forward(INPUT_IDS)
    model.close()

    assert not torch.allclose(baseline_logits, perturbed_logits)


def test_perturbing_the_router_scale_changes_the_output(tmp_path):
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)
    with torch.no_grad():
        baseline_logits = model.forward(INPUT_IDS)
        for layer in model.layers:
            layer.moe.router.scale.add_(1.0)
        perturbed_logits = model.forward(INPUT_IDS)
    model.close()

    assert not torch.allclose(baseline_logits, perturbed_logits)


def test_dense_gemma4_checkpoint_gets_no_moe_block(tmp_path):
    path = build_tiny_gemma4_gguf(tmp_path / "tiny-gemma4-dense.gguf")
    model = _load(path)

    assert not model.is_moe
    for layer in model.layers:
        assert layer.moe is None


def test_layer_count_matches_the_real_fixture(tmp_path):
    path = build_tiny_gemma4_moe_gguf(tmp_path / "tiny-gemma4-moe.gguf")
    model = _load(path)

    assert len(model.layers) == MOE_N_LAYER
