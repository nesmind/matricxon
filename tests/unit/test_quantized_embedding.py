"""QuantizedEmbedding (app/architectures/quantized_embedding.py) - the shared "keep this table
packed, dequantize only the rows a forward pass actually looks up" module every tied-embedding
architecture (llama, mistral3, now gemma4) composes. Exercised directly (no GGUF file, no full
architecture) with a small synthetic Q4_K table, both backends (see the module-scoped `numba`/
`native` fixtures) cross-checked against each other and against `as_linear()`'s own tied output
projection.
"""

import random
import shutil
import struct

import pytest
import torch

from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.native.gemm import NativeGemm
from app.native.library import NativeKernelLibrary

NUM_EMBEDDINGS = 6
EMBEDDING_DIM = 256  # one Q4_K block per row


def _q4_k_table(n_rows: int, seed: int) -> bytes:
    rng = random.Random(seed)

    def block() -> bytes:
        d = struct.pack("<e", rng.uniform(0.01, 1.0))
        dmin = struct.pack("<e", rng.uniform(0.01, 1.0))
        scales_and_qs = bytes(rng.randrange(256) for _ in range(12 + 128))
        return d + dmin + scales_and_qs

    return b"".join(block() for _ in range(n_rows))


@pytest.fixture(autouse=True)
def numba_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NativeGemm, "_active", None)


def test_forward_returns_the_real_dequantized_row_per_token() -> None:
    raw = _q4_k_table(NUM_EMBEDDINGS, seed=0)
    embedding = QuantizedEmbedding(NUM_EMBEDDINGS, EMBEDDING_DIM, T.Q4_K, memoryview(raw))
    input_ids = torch.tensor([[3, 0, 3]])

    result = embedding(input_ids)

    strategy = QuantStrategyRegistry().get(T.Q4_K)
    row_bytes = len(raw) // NUM_EMBEDDINGS
    for t, token_id in enumerate([3, 0, 3]):
        start = token_id * row_bytes
        expected = strategy.dequantize(memoryview(raw[start : start + row_bytes]), EMBEDDING_DIM)
        assert torch.allclose(result[0, t], expected, atol=1e-4)


def test_as_linear_matches_a_plain_dequantized_matmul() -> None:
    raw = _q4_k_table(NUM_EMBEDDINGS, seed=1)
    embedding = QuantizedEmbedding(NUM_EMBEDDINGS, EMBEDDING_DIM, T.Q4_K, memoryview(raw))
    lm_head = embedding.as_linear()
    assert isinstance(lm_head, QuantizedLinear)
    x = torch.randn(1, 1, EMBEDDING_DIM)

    result = lm_head(x)

    strategy = QuantStrategyRegistry().get(T.Q4_K)
    weight = strategy.dequantize(memoryview(raw), NUM_EMBEDDINGS * EMBEDDING_DIM)
    weight = weight.reshape(NUM_EMBEDDINGS, EMBEDDING_DIM)
    expected = x.reshape(1, EMBEDDING_DIM) @ weight.T
    assert torch.allclose(result.reshape(1, NUM_EMBEDDINGS), expected, atol=1e-3)


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs a C compiler")
def test_native_backend_matches_the_python_fallback(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _q4_k_table(NUM_EMBEDDINGS, seed=2)
    input_ids = torch.tensor([[5, 5, 1, 4]])

    dense = QuantizedEmbedding(NUM_EMBEDDINGS, EMBEDDING_DIM, T.Q4_K, memoryview(raw))
    expected = dense(input_ids)

    lib = NativeKernelLibrary(build_dir=tmp_path_factory.mktemp("native_build")).load()
    monkeypatch.setattr(NativeGemm, "_active", NativeGemm(lib, n_threads=2))
    packed = QuantizedEmbedding(NUM_EMBEDDINGS, EMBEDDING_DIM, T.Q4_K, memoryview(raw))
    assert (
        packed._native is not None
    )  # actually exercising the native path, not silently falling back

    result = packed(input_ids)

    assert torch.allclose(result, expected, atol=1e-4)
