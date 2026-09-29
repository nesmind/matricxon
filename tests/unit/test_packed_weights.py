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
from app.architectures.nemotron_h import NemotronHArchitecture
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
from tests.tiny_gguf_nemotron_h_q8 import LAYER_TYPES, build_tiny_nemotron_h_q8_gguf

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


class TestPackedNemotronH:
    """attn_v/attn_output/ffn_up/ffn_down/output.weight wiring added 2026-09-29 - see
    `app/models/load_dtype.py`'s own `_QUANTIZED_NATIVE_TENSOR_SUFFIXES_BY_ARCH["nemotron_h"]`
    comment for why (real, non-marginal RAM savings on an installed 12B checkpoint, not the
    "marginal" call this repo's own comment used to make) - including the Mamba mixer's own
    `in_proj`/`out_proj` (real `ssm_in`/`ssm_out` tensors, plain 2D `nn.Linear` despite living in
    the "SSM block"). attn_q/attn_k/token_embd/every other SSM tensor are never eligible (see
    `NemotronHArchitecture`'s own docstring) and must stay plain `nn.Linear`/`nn.Parameter`/
    `nn.Embedding` either way - checked below alongside the ones that do become `QuantizedLinear`.
    """

    @pytest.fixture
    def gguf_path(self, tmp_path: Path) -> Path:
        return build_tiny_nemotron_h_q8_gguf(tmp_path / "tiny-nemotron-h-q8.gguf")

    def _load(self, path: Path, packed: bool) -> NemotronHArchitecture:
        loader = GGUFModelLoader(path, dtype=torch.float32)
        return NemotronHArchitecture.from_gguf(
            loader, dtype=torch.float32, enable_quantized_native=packed
        )

    def test_only_the_eligible_projections_become_quantized_linear(self, gguf_path: Path) -> None:
        model = self._load(gguf_path, packed=True)
        with torch.no_grad():
            # Materialization is deferred until the first real forward pass (see
            # ModelArchitecture._ensure_materialized) - the placeholder nn.Linear built in
            # __init__ never becomes QuantizedLinear until this runs.
            cache = model.build_cache(max_seq_len=4, dtype=torch.float32)
            model.forward(torch.tensor([[1, 2]]), cache)

        assert not isinstance(model.token_embd, QuantizedEmbedding)
        assert isinstance(model.lm_head, QuantizedLinear)
        for i, layer_type in enumerate(LAYER_TYPES):
            layer = model.layers[i]
            if layer_type == "attention":
                assert not isinstance(layer.self_attn.q_proj, QuantizedLinear)
                assert not isinstance(layer.self_attn.k_proj, QuantizedLinear)
                assert isinstance(layer.self_attn.v_proj, QuantizedLinear)
                assert isinstance(layer.self_attn.o_proj, QuantizedLinear)
            elif layer_type == "mlp":
                assert isinstance(layer.mlp.up_proj, QuantizedLinear)
                assert isinstance(layer.mlp.down_proj, QuantizedLinear)
            else:
                assert isinstance(layer.mixer.in_proj, QuantizedLinear)
                assert isinstance(layer.mixer.out_proj, QuantizedLinear)
        model.close()

    def test_packed_model_matches_float_model(self, gguf_path: Path) -> None:
        dense = self._load(gguf_path, packed=False)
        packed = self._load(gguf_path, packed=True)
        input_ids = torch.tensor([[1, 5, 9, 20]])
        with torch.no_grad():
            dense_cache = dense.build_cache(max_seq_len=8, dtype=torch.float32)
            packed_cache = packed.build_cache(max_seq_len=8, dtype=torch.float32)
            expected = dense.forward(input_ids, dense_cache)
            result = packed.forward(input_ids, packed_cache)

        assert torch.allclose(result, expected, atol=1e-4)
        packed.close()
        dense.close()


def test_ram_estimate_counts_packed_projections_for_nemotron_h(tmp_path: Path) -> None:
    gguf_path = build_tiny_nemotron_h_q8_gguf(tmp_path / "tiny-nemotron-h-q8.gguf")
    tensor_infos = GGUFReader(gguf_path).read().tensor_infos
    packed_names = (
        "attn_v.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
        "output.weight",
        "ssm_in.weight",
        "ssm_out.weight",
    )
    registry = QuantStrategyRegistry()
    saved = sum(
        t.n_elements * 2 - registry.get(t.ggml_type).byte_length(t.n_elements)
        for t in tensor_infos
        if t.name.endswith(packed_names)
    )
    without = estimate_quantized_native_bytes(tensor_infos, enabled=True, architecture_name="")
    with_nemotron_h = estimate_quantized_native_bytes(tensor_infos, True, "nemotron_h")

    assert saved > 0
    assert with_nemotron_h <= without - saved
