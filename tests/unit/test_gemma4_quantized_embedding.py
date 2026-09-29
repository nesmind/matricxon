"""gemma4's own real, confirmed gap (2026-09-30): token_embd.weight (its tied lm_head too - no
separate output.weight ever exists, see Gemma4Architecture's own docstring) and
per_layer_token_embd.weight (real Per-Layer Embeddings, ~50% of a real gemma-4-E2B-it checkpoint's
own elements - see gemma4_ple.py) were never routed through QuantizedEmbedding at all, unlike
llama/mistral3's identical tied-embedding case - always the full dequantized table regardless of
Settings.enable_quantized_native_compute. Exercised on tests/tiny_gguf_gemma4_q8.py's tiny
Q8_0 fixture, packed vs dense compared on the exact same quantized values.
"""

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
    # gemma4 always ties (see this class's own docstring) - a packed table must always get its
    # tied lm_head via as_linear(), never left calling F.linear on a table that no longer has a
    # real dense .weight tensor at all.
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
    """Every real packed QuantizedEmbedding must be a discoverable submodule (see
    PackedWeightLoading._release_packed_modules) - otherwise ModelArchitecture.close()'s own walk
    would miss it, and mmap.close() would raise a real BufferError instead of this test just
    passing."""
    path = build_tiny_gemma4_ple_q8_gguf(tmp_path / "tiny-gemma4-ple-q8.gguf")
    model = _load(path, packed=True)
    with torch.no_grad():
        model.forward(INPUT_IDS)

    model.close()  # must not raise
