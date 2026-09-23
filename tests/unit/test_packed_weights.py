"""Packed (still quantized) weights beyond the per-layer projections: PackedRows, the packed
token-embedding table (QuantizedEmbedding, also the tied lm_head), attn_q/attn_k with their
`unpermute_rope_rows` applied to packed rows, and `last_logits_only`. Each packed path is checked
against the float path on the exact same quantized file (tests/tiny_gguf_llama_q8.py).
"""

from pathlib import Path

import pytest
import torch

from app.architectures.layers import unpermute_rope_rows
from app.architectures.llama import LlamaArchitecture
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.quantized_linear import QuantizedLinear
from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.loader import GGUFModelLoader
from app.gguf.packed_rows import PackedRows
from app.gguf.reader import GGUFReader
from app.models.load_dtype import estimate_quantized_native_bytes
from app.native.gemm import NativeGemm
from tests.tiny_gguf_llama_q8 import N_HEAD, _q8_0_bytes, build_tiny_llama_q8_gguf

INPUT_IDS = torch.tensor([[1, 72, 105, 33, 90]])


@pytest.fixture(autouse=True)
def numba_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(NativeGemm, "_active", None)


@pytest.fixture
def gguf_path(tmp_path: Path) -> Path:
    return build_tiny_llama_q8_gguf(tmp_path / "tiny-llama-q8.gguf")


def _load(path: Path, packed: bool) -> LlamaArchitecture:
    loader = GGUFModelLoader(path, dtype=torch.float32)
    return LlamaArchitecture.from_gguf(loader, dtype=torch.float32, enable_quantized_native=packed)


def _dequantize(raw: memoryview, rows: int, cols: int) -> torch.Tensor:
    return QuantStrategyRegistry().get(T.Q8_0).dequantize(raw, rows * cols).reshape(rows, cols)


class TestPackedRows:
    def test_reordered_matches_unpermute_on_dequantized_weight(self) -> None:
        weight = torch.randn(64, 64)
        raw = memoryview(_q8_0_bytes(weight.numpy()))
        order = unpermute_rope_rows(torch.arange(64), N_HEAD)

        reordered = PackedRows(raw, 64).reordered(order)

        expected = unpermute_rope_rows(_dequantize(raw, 64, 64), N_HEAD)
        assert torch.equal(_dequantize(reordered, 64, 64), expected)

    def test_rejects_bytes_that_dont_split_into_rows(self) -> None:
        with pytest.raises(ValueError):
            PackedRows(memoryview(bytes(10)), 3)


class TestQuantizedEmbedding:
    def test_lookup_and_tied_linear_match_the_dense_table(self) -> None:
        table = torch.randn(10, 64)
        raw = memoryview(_q8_0_bytes(table.numpy()))
        dense = _dequantize(raw, 10, 64)
        embedding = QuantizedEmbedding(10, 64, T.Q8_0, raw)
        ids = torch.tensor([[3, 0, 3, 9]])

        assert torch.equal(embedding(ids), dense[ids])
        x = torch.randn(1, 2, 64)
        assert torch.allclose(embedding.as_linear()(x), x @ dense.T, atol=1e-4)


class TestPackedLlama:
    def test_packed_model_matches_float_model(self, gguf_path: Path) -> None:
        dense = _load(gguf_path, packed=False)
        packed = _load(gguf_path, packed=True)
        with torch.no_grad():
            expected = dense.forward(INPUT_IDS)
            result = packed.forward(INPUT_IDS)

        assert isinstance(packed.token_embd, QuantizedEmbedding)
        assert isinstance(packed.lm_head, QuantizedLinear)
        assert isinstance(packed.layers[0].self_attn.q_proj, QuantizedLinear)
        assert isinstance(packed.layers[0].self_attn.k_proj, QuantizedLinear)
        assert torch.allclose(result, expected, atol=1e-4)
        packed.close()
        dense.close()

    @pytest.mark.parametrize("packed", [False, True])
    def test_last_logits_only_is_the_last_row_of_the_full_logits(
        self, gguf_path: Path, packed: bool
    ) -> None:
        model = _load(gguf_path, packed=packed)
        with torch.no_grad():
            full = model.forward(INPUT_IDS)
            last = model.forward(INPUT_IDS, last_logits_only=True)

        assert last.shape == (1, 1, full.shape[-1])
        assert torch.allclose(last, full[:, -1:, :], atol=1e-5)
        model.close()


def test_ram_estimate_counts_packed_qk_and_embedding_for_llama(gguf_path: Path) -> None:
    tensor_infos = GGUFReader(gguf_path).read().tensor_infos
    packed_names = ("attn_q.weight", "attn_k.weight", "token_embd.weight")
    registry = QuantStrategyRegistry()
    saved = sum(
        t.n_elements * 2 - registry.get(t.ggml_type).byte_length(t.n_elements)
        for t in tensor_infos
        if t.name.endswith(packed_names)
    )
    without = estimate_quantized_native_bytes(tensor_infos, enabled=True, architecture_name="")
    with_llama = estimate_quantized_native_bytes(tensor_infos, True, "llama")

    assert saved > 0
    assert with_llama <= without - saved
