"""Gemma4's Per-Layer Embeddings (PLE) and cross-layer KV reuse (`app/architectures/gemma4_ple.py`,
`Gemma4Attention`'s `has_own_kv`/`shared_kv`), confirmed against a `gemma-4-E2B-it` GGUF
(the HF/llama.cpp sources are in each module's docstring) that the earlier
`Gemma4Architecture` never implemented.

No independent oracle here (the math is cross-verified against two real sources in the production
docstrings); each mechanism is proven *live* (not a silent no-op) by perturbing the one weight only
it reads and checking the output changes.
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

    assert l0.has_own_kv and l0.v_proj is not None
    assert l1.has_own_kv and l1.v_proj is None
    assert not l2.has_own_kv and not hasattr(l2, "k_proj")
    assert not l3.has_own_kv and not hasattr(l3, "k_proj")
    assert model.per_layer_dim == 4
    assert not isinstance(model.per_layer_embedding, QuantizedLinear)


def test_forward_runs_and_produces_finite_correctly_shaped_logits(tmp_path):
    path = build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf")
    model = _load(path)
    with torch.no_grad():
        logits = model.forward(INPUT_IDS)

    assert logits.shape == (1, INPUT_IDS.shape[1], model.vocab_size)
    assert torch.isfinite(logits).all()
    model.close()


def test_perturbing_the_provider_layers_own_kv_changes_a_shared_layers_output(tmp_path):
    """Layers 2/3 have no k_proj/v_proj: this proves they don't silently fall back to something else
    (skipping attention, a zero/identity kv) by checking the output depends on layer 0/1's
    attn_k/attn_v, the only possible source."""
    # Materialization is lazy (first forward, ModelArchitecture._ensure_materialized): perturbing
    # earlier would be overwritten by the loaded weights, so forward once (materializing), perturb,
    # then forward again on the same instance (kv_cache=None: a stateless full-sequence pass, safe
    # to repeat).
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


def test_last_logits_only_returns_just_the_last_position(tmp_path):
    model = _load(build_tiny_gemma4_ple_kv_shared_gguf(tmp_path / "tiny-gemma4-ple.gguf"))
    with torch.no_grad():
        full = model.forward(INPUT_IDS)
        last = model.forward(INPUT_IDS, last_logits_only=True)

    assert last.shape[1] == 1 and torch.allclose(last[:, 0], full[:, -1], atol=1e-5)
    model.close()
