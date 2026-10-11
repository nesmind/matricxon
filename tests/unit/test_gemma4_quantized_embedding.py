"""gemma4's gap: token_embd.weight (also its tied lm_head; no output.weight exists,
Gemma4Architecture) and per_layer_token_embd.weight (Per-Layer Embeddings, ~50% of a gemma-4-E2B-it
checkpoint's elements; gemma4_ple.py) never went through QuantizedEmbedding, unlike llama/mistral3's
tied-embedding case: always the full dequantized table whatever
Settings.enable_quantized_native_compute. Exercised on tests/tiny_gguf_gemma4_q8.py's Q8_0 fixture,
packed vs dense on the same quantized values."""

import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf_gemma4_q8 import build_tiny_gemma4_ple_q8_gguf

INPUT_IDS = torch.tensor([[4, 5, 4, 6]])


def _load(path, packed: bool) -> Gemma4Architecture:
    loader = GGUFModelLoader(path, dtype=torch.float32)
    return Gemma4Architecture.from_gguf(loader, dtype=torch.float32, enable_quantized_native=packed)


def test_packed_model_matches_the_dense_model(tmp_path):
    path = build_tiny_gemma4_ple_q8_gguf(tmp_path / "tiny-gemma4-ple-q8.gguf")
    dense = _load(path, packed=False)
    packed = _load(path, packed=True)
    with torch.no_grad():
        expected = dense.forward(INPUT_IDS)
        result = packed.forward(INPUT_IDS)

    assert torch.allclose(result, expected, atol=1e-3)
    packed.close()
    dense.close()


def test_packed_model_builds_a_quantized_embedding_for_both_tables(tmp_path):
    path = build_tiny_gemma4_ple_q8_gguf(tmp_path / "tiny-gemma4-ple-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    assert isinstance(model.token_embd, QuantizedEmbedding)
    assert isinstance(model.per_layer_embedding.embed_tokens_per_layer, QuantizedEmbedding)
    # gemma4 always ties (class docstring): a packed table must get its tied lm_head via
    # as_linear, never F.linear on a table with no dense .weight.
    assert hasattr(model, "lm_head")
    model.close()


def test_dense_model_never_builds_any_quantized_embedding(tmp_path):
    path = build_tiny_gemma4_ple_q8_gguf(tmp_path / "tiny-gemma4-ple-q8.gguf")
    model = _load(path, packed=False)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    assert not isinstance(model.token_embd, QuantizedEmbedding)
    assert not isinstance(model.per_layer_embedding.embed_tokens_per_layer, QuantizedEmbedding)
    assert not hasattr(model, "lm_head")
    model.close()


def test_release_lets_the_mmap_close_cleanly(tmp_path):
    """Every packed QuantizedEmbedding must be a discoverable submodule
    (PackedWeightLoading._release_packed_modules); otherwise ModelArchitecture.close's walk misses
    it and mmap.close raises BufferError."""
    path = build_tiny_gemma4_ple_q8_gguf(tmp_path / "tiny-gemma4-ple-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    model.close()
