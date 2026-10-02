"""gemma4 (PLE, sliding masks, MoE-capable) and granitemoe on the device path, without a GPU:
packed placement on the CPU device must reproduce the plain model's logits, with every expert
and the packed embedding tables actually held as `DevicePacked*` modules."""

from pathlib import Path

import pytest
import torch

from app.architectures.device_packed import DevicePackedEmbedding, DevicePackedLinear
from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.granitemoe import GraniteMoeArchitecture
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf_gemma4_q8 import build_tiny_gemma4_ple_q8_gguf
from tests.tiny_gguf_granitemoe_q8 import build_tiny_granitemoe_q8_gguf

CASES = {
    "gemma4": (build_tiny_gemma4_ple_q8_gguf, Gemma4Architecture),
    "granitemoe": (build_tiny_granitemoe_q8_gguf, GraniteMoeArchitecture),
}


def _load(path: Path, cls: type, packed: bool):
    model = cls.from_gguf(
        GGUFModelLoader(path, dtype=torch.float32),
        dtype=torch.float32,
        enable_quantized_native=packed,
    )
    model.place_on(torch.device("cpu"), packed_weights=packed)
    return model.eval()


@pytest.mark.parametrize("name", sorted(CASES))
def test_packed_placement_matches_the_plain_model(name: str, tmp_path: Path) -> None:
    builder, cls = CASES[name]
    path = builder(tmp_path / "m.gguf")
    plain, packed = _load(path, cls, False), _load(path, cls, True)
    ids = torch.tensor([[1, 5, 9, 3, 7]])
    with torch.no_grad():
        expected, actual = plain(ids), packed(ids)
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    kinds = {type(m) for m in packed.modules()}
    assert DevicePackedLinear in kinds
    if name == "gemma4":  # granitemoe never routed token_embd through a packed table
        assert DevicePackedEmbedding in kinds


def test_packed_moe_experts_are_per_expert_device_modules(tmp_path: Path) -> None:
    builder, cls = CASES["granitemoe"]
    model = _load(builder(tmp_path / "m.gguf"), cls, True)
    model(torch.tensor([[1, 2, 3]]))
    experts = model.layers[0].mlp.experts
    assert len(experts.gate_packed) == experts.num_experts
    assert all(isinstance(m, DevicePackedLinear) for m in experts.gate_packed.values())
    assert experts.gate_exps.numel() == 0  # the dense placeholder was dropped
