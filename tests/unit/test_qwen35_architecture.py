"""`Qwen35Architecture` against a tiny synthetic `qwen35` GGUF: registry resolution, layer
pattern, and that incremental decoding (hybrid cache) matches a single full-sequence pass.
"""

from pathlib import Path

import torch

from app.architectures.qwen35 import Qwen35Architecture
from app.architectures.registry import ArchitectureRegistry
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from tests.tiny_gguf_qwen35 import N_LAYER, build_tiny_qwen35_gguf


def _model(tmp_path: Path, tied: bool = False) -> Qwen35Architecture:
    path = build_tiny_qwen35_gguf(tmp_path / "t.gguf", tied_embeddings=tied)
    model = Qwen35Architecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    return model.eval()


def test_registry_resolves_qwen35(tmp_path: Path) -> None:
    path = build_tiny_qwen35_gguf(tmp_path / "t.gguf")
    metadata = GGUFReader(path).read().metadata
    assert ArchitectureRegistry().resolve(metadata) is Qwen35Architecture
    assert "qwen35" in ArchitectureRegistry().supported_names()


def test_layer_pattern_is_three_linear_then_one_attention(tmp_path: Path) -> None:
    model = _model(tmp_path)
    assert model.layer_types == ["mamba", "mamba", "mamba", "attention"][:N_LAYER]


@torch.no_grad()
def test_incremental_decode_matches_full_pass(tmp_path: Path) -> None:
    model = _model(tmp_path)
    ids = torch.tensor([[3, 40, 77, 120, 9, 200, 31]])
    full = model(ids)

    cache = model.build_cache(max_seq_len=16, dtype=torch.float32)
    steps = [model(ids[:, :4], cache, torch.arange(4))]
    cache.advance(4)
    for t in range(4, 7):
        steps.append(model(ids[:, t : t + 1], cache, torch.tensor([t])))
        cache.advance(1)
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-4)


@torch.no_grad()
def test_tied_embeddings_forward_runs(tmp_path: Path) -> None:
    out = _model(tmp_path, tied=True)(torch.tensor([[3, 4, 5]]))
    assert out.shape[:2] == (1, 3)
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_last_logits_only_returns_just_the_last_position(tmp_path: Path) -> None:
    model = _model(tmp_path)
    ids = torch.tensor([[3, 40, 77, 120]])
    full = model(ids)
    last = model(ids, last_logits_only=True)
    assert last.shape[1] == 1 and torch.allclose(last[:, 0], full[:, -1], atol=1e-5)
