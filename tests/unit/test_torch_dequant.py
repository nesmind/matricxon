"""The torch dequantizers (on-device, `gpu_weight_mode="packed"`) must equal the numpy
`QuantStrategy` of the same type exactly, and a model loaded with packed weights must produce the
same logits as the dequantized one. Packed mode also runs on the CPU device, so no GPU is needed."""

from pathlib import Path

import numpy as np
import pytest
import torch

from app.architectures.device_packed import DevicePackedEmbedding, DevicePackedLinear
from app.architectures.llama import LlamaArchitecture
from app.gguf.constants import GGMLQuantizationType as T
from app.gguf.dequant.registry import QuantStrategyRegistry
from app.gguf.dequant.torch_dequant import TorchDequantizer
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf_llama_q8 import build_tiny_llama_q8_gguf

# Byte offsets of each type's f16 fields (random bytes would give NaN/inf scales).
F16_FIELDS = {
    T.Q8_0: [0], T.Q4_0: [0], T.Q4_1: [0, 2], T.Q5_0: [0], T.Q5_1: [0, 2],
    T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208], T.Q2_K: [80, 82], T.Q3_K: [108],
    T.Q8_K: [], T.IQ4_NL: [0], T.IQ4_XS: [0],
}  # fmt: skip


def _random_blocks(ggml_type: int, n_blocks: int) -> np.ndarray:
    rng = np.random.RandomState(7)
    _, type_size = TorchDequantizer.geometry(ggml_type)
    blocks = rng.randint(0, 256, size=(n_blocks, type_size)).astype(np.uint8)
    for offset in F16_FIELDS[ggml_type]:
        halves = (rng.randn(n_blocks) * 0.1).astype("<f2")
        blocks[:, offset : offset + 2] = halves.view(np.uint8).reshape(n_blocks, 2)
    return blocks


@pytest.mark.parametrize("ggml_type", sorted(F16_FIELDS), ids=lambda t: T(t).name)
def test_matches_the_numpy_strategy(ggml_type: int) -> None:
    blocks = _random_blocks(ggml_type, 5)
    block_size, _ = TorchDequantizer.geometry(ggml_type)
    expected = (
        QuantStrategyRegistry()
        .get(ggml_type)
        .dequantize(memoryview(blocks.tobytes()), 5 * block_size)
    )
    actual = TorchDequantizer.dequantize(torch.from_numpy(blocks), ggml_type)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_unsupported_types_are_reported() -> None:
    assert not TorchDequantizer.supports(T.IQ2_XS) and TorchDequantizer.supports(T.Q4_K)


@pytest.fixture
def gguf_path(tmp_path: Path) -> Path:
    return build_tiny_llama_q8_gguf(tmp_path / "tiny-llama-q8.gguf")


def _load(path: Path, packed: bool) -> LlamaArchitecture:
    loader = GGUFModelLoader(path, dtype=torch.float32)
    model = LlamaArchitecture.from_gguf(loader, dtype=torch.float32, enable_quantized_native=packed)
    model.place_on(torch.device("cpu"), packed_weights=packed)
    return model.eval()


class TestPackedModel:
    def test_logits_match_the_dequantized_model(self, gguf_path: Path) -> None:
        ids = torch.tensor([[1, 72, 105, 33, 90]])
        dense, packed = _load(gguf_path, False), _load(gguf_path, True)
        with torch.no_grad():
            expected, actual = dense(ids), packed(ids)
        assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)

    def test_projections_and_embedding_stay_packed(self, gguf_path: Path) -> None:
        packed = _load(gguf_path, True)
        packed(torch.tensor([[1, 2]]))
        kinds = {type(m) for m in packed.modules()}
        assert DevicePackedLinear in kinds and DevicePackedEmbedding in kinds
        assert packed._quantized_loader is None  # no mmap kept open: bytes live on the device

    def test_chunked_rows_match_a_single_pass(self, monkeypatch, gguf_path: Path) -> None:
        import app.architectures.device_packed as device_packed

        packed = _load(gguf_path, True)
        ids = torch.tensor([[1, 72, 105]])
        whole = packed(ids)
        monkeypatch.setattr(device_packed, "_CHUNK_ELEMENTS", 64 * 8)  # forces many row slices
        assert torch.allclose(packed(ids), whole, atol=1e-5)
