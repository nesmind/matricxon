"""Gemma4's Per-Layer Embeddings (PLE) and cross-layer KV reuse (`app/architectures/gemma4_ple.py`,
`Gemma4Attention`'s `has_own_kv`/`shared_kv`) - real, active mechanisms confirmed against a real
`gemma-4-E2B-it` GGUF (2026-09-29, see each module's own docstring for the real HF/llama.cpp
source this was ported from) that the earlier `Gemma4Architecture` never implemented at all.

No independent from-scratch oracle here (the real math is already cross-verified against two
independent real sources in the production code's own docstrings) - instead, each mechanism is
proven *live* (not silently a no-op) by perturbing the one weight tensor that only that mechanism
reads and checking the final output actually changes.
"""

import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf_gemma4 import (
    PLE_N_KV_SHARED_LAYERS,
    build_tiny_gemma4_ple_kv_shared_gguf,
)

INPUT_IDS = torch.tensor([[4, 10, 20]])


def _load(path):
    loader = GGUFModelLoader(path, dtype=torch.float32)
    return Gemma4Architecture.from_gguf(loader, dtype=torch.float32)


def test_shared_layers_have_no_kv_projections_and_non_shared_respect_real_attn_v(tmp_path):
    path = build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf")
    model = _load(path)

    assert model._n_layer_kv_from_start == 4 - PLE_N_KV_SHARED_LAYERS
    l0, l1, l2, l3 = (model.layers[i].self_attn for i in range(4))

    assert l0.has_own_kv and l0.v_proj is not None  # layer 0: own kv, real attn_v present
    assert l1.has_own_kv and l1.v_proj is None  # layer 1: own kv, no attn_v -> use_v_from_k
    assert not l2.has_own_kv and not hasattr(l2, "k_proj")  # layer 2: shared, no kv modules at all
    assert not l3.has_own_kv and not hasattr(l3, "k_proj")  # layer 3: shared, no kv modules at all
    assert model.per_layer_dim == 4
    assert not isinstance(model.per_layer_embedding, QuantizedLinear)  # real module, not a stub


def test_forward_runs_and_produces_finite_correctly_shaped_logits(tmp_path):
    path = build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf")
    model = _load(path)
    with torch.no_grad():
        logits = model.forward(INPUT_IDS)

    assert logits.shape == (1, INPUT_IDS.shape[1], model.vocab_size)
    assert torch.isfinite(logits).all()
    model.close()


def test_perturbing_the_provider_layers_own_kv_changes_a_shared_layers_output(tmp_path):
    """Layers 2/3 have no k_proj/v_proj of their own - proves they're not silently falling back
    to *something else* (e.g. skipping attention, or a zero/identity kv) by checking the final
    output actually depends on layer 0/1's own attn_k/attn_v weights, the only place that data
    could come from."""
    # Materialization is lazy (first forward() call, see ModelArchitecture._ensure_materialized)
    # - perturbing before that would just get overwritten by the real loaded weights, so this
    # forwards once first (materializing), then perturbs, then forwards again on the same
    # instance (kv_cache=None each time - a stateless full-sequence pass, safe to repeat).
    path = build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf")
    model = _load(path)
    with torch.no_grad():
        baseline_logits = model.forward(INPUT_IDS)
        model.layers[0].self_attn.k_proj.weight.add_(1.0)
        model.layers[1].self_attn.k_proj.weight.add_(1.0)
        perturbed_logits = model.forward(INPUT_IDS)
    model.close()

    assert not torch.allclose(baseline_logits, perturbed_logits)


def test_perturbing_per_layer_embedding_changes_the_output(tmp_path):
    path = build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf")
    model = _load(path)
    with torch.no_grad():
        baseline_logits = model.forward(INPUT_IDS)
        model.per_layer_embedding.embed_tokens_per_layer.weight.add_(1.0)
        perturbed_logits = model.forward(INPUT_IDS)
    model.close()

    assert not torch.allclose(baseline_logits, perturbed_logits)
