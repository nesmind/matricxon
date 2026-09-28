"""ClipVisionEncoder against a tiny but real Idefics3/SmolVLM-shaped mmproj

GGUF (see tests/tiny_gguf_idefics3.py) - fast, memory-safe wiring proof for
the `idefics3` projector path (pixel-shuffle merge + a single bias-free
linear), mirroring test_clip_vision_encoder.py's MLP-path coverage. The
real numeric/statistical validation against the real
`ggml-org/SmolVLM-256M-Instruct-GGUF` mmproj pull is done manually (see
ROADMAP.md's vision-fusion entry).
"""

from pathlib import Path

import pytest
import torch

from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.vision.clip_vision_encoder import ClipVisionEncoder
from tests.tiny_gguf_idefics3 import (
    IMAGE_SIZE,
    PROJECTION_DIM,
    SCALE_FACTOR,
    build_tiny_idefics3_gguf,
)


@pytest.fixture
def tiny_idefics3_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_idefics3.gguf"
    build_tiny_idefics3_gguf(path)
    return path


class TestSupports:
    def test_recognizes_a_real_idefics3_clip_vision_gguf(self, tiny_idefics3_path: Path) -> None:
        metadata = GGUFReader(tiny_idefics3_path).read().metadata
        assert ClipVisionEncoder.supports(metadata) is True


class TestFromGguf:
    def test_num_patches_is_the_raw_grid_divided_by_scale_factor_squared(
        self, tiny_idefics3_path: Path
    ) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        raw_grid = (IMAGE_SIZE // model.patch_size) ** 2
        assert model.raw_num_patches == raw_grid
        assert model.num_patches == raw_grid // (SCALE_FACTOR * SCALE_FACTOR)

    def test_builds_a_bias_free_projector(self, tiny_idefics3_path: Path) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        assert model.mm_fc is not None
        assert model.mm_fc.bias is None
        assert model.projector_up is None
        assert model.projector_down is None


class TestForward:
    def test_produces_the_expected_output_shape(self, tiny_idefics3_path: Path) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(pixel_values)

        assert out.shape == (1, model.num_patches, PROJECTION_DIM)

    def test_output_has_no_nan_or_inf(self, tiny_idefics3_path: Path) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(pixel_values)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_produce_different_outputs(self, tiny_idefics3_path: Path) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        a = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        b = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out_a = model(a)
            out_b = model(b)

        assert not torch.allclose(out_a, out_b)

    def test_same_input_is_deterministic(self, tiny_idefics3_path: Path) -> None:
        loader = GGUFModelLoader(tiny_idefics3_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            first = model(pixel_values)
            second = model(pixel_values)

        assert torch.equal(first, second)
