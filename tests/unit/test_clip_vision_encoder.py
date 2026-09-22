"""M10: ClipVisionEncoder against a tiny but real `clip` mmproj GGUF (see

tests/tiny_gguf_clip.py) - fast, memory-safe wiring proof. The real
numeric/statistical validation against the real moondream2 mmproj pull
(910MB) was done manually (see ROADMAP.md's M10 vision entry) - this
covers shape correctness, the real ffn_up/ffn_down naming swap, and
end-to-end no-crash/no-NaN behavior.
"""

from pathlib import Path

import pytest
import torch

from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.vision.clip_vision_encoder import ClipVisionEncoder
from app.vision.image_preprocessing import ClipImagePreprocessor
from tests.tiny_gguf_clip import IMAGE_SIZE, N_EMBD, PROJECTION_DIM, build_tiny_clip_gguf


@pytest.fixture
def tiny_clip_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_clip.gguf"
    build_tiny_clip_gguf(path)
    return path


class TestSupports:
    def test_recognizes_a_real_clip_vision_gguf(self, tiny_clip_path: Path) -> None:
        metadata = GGUFReader(tiny_clip_path).read().metadata
        assert ClipVisionEncoder.supports(metadata) is True

    def test_rejects_a_text_architecture(self, tiny_clip_path: Path) -> None:
        metadata = GGUFReader(tiny_clip_path).read().metadata
        assert metadata.architecture == "clip"
        # A real clip file with no vision encoder (has_vision_encoder=False,
        # e.g. a text-only CLIP export) must not be mistaken for one.
        from app.gguf.metadata import GGUFMetadata

        text_only = GGUFMetadata({**metadata._values, "clip.has_vision_encoder": False})
        assert ClipVisionEncoder.supports(text_only) is False


class TestForward:
    def test_produces_the_expected_output_shape(self, tiny_clip_path: Path) -> None:
        loader = GGUFModelLoader(tiny_clip_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(pixel_values)

        assert out.shape == (1, model.num_patches, PROJECTION_DIM)

    def test_output_has_no_nan_or_inf(self, tiny_clip_path: Path) -> None:
        loader = GGUFModelLoader(tiny_clip_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out = model(pixel_values)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_produce_different_outputs(self, tiny_clip_path: Path) -> None:
        loader = GGUFModelLoader(tiny_clip_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        a = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        b = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            out_a = model(a)
            out_b = model(b)

        assert not torch.allclose(out_a, out_b)

    def test_same_input_is_deterministic(self, tiny_clip_path: Path) -> None:
        loader = GGUFModelLoader(tiny_clip_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            first = model(pixel_values)
            second = model(pixel_values)

        assert torch.equal(first, second)

    def test_ffn_layers_use_the_real_confirmed_shapes(self, tiny_clip_path: Path) -> None:
        """Regression guard for the real, confirmed-by-bias-length finding

        that "ffn_up"/"ffn_down" naming in this GGUF format is the
        opposite of what the names suggest (see ClipVisionEncoderLayer's
        own docstring) - fc1 must expand to ffn_len, fc2 must contract
        back to n_embd, regardless of which GGUF tensor name feeds which.
        """
        loader = GGUFModelLoader(tiny_clip_path, dtype=torch.float32)
        model = ClipVisionEncoder.from_gguf(loader)

        layer = model.layers[0]
        assert layer.fc1.in_features == N_EMBD
        assert layer.fc1.out_features == model.ffn_len
        assert layer.fc2.in_features == model.ffn_len
        assert layer.fc2.out_features == N_EMBD


class TestImagePreprocessor:
    def test_preprocesses_a_real_png_to_the_expected_shape(self) -> None:
        import base64
        import io

        from PIL import Image

        image = Image.new("RGB", (32, 32), (255, 0, 0))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

        preprocessor = ClipImagePreprocessor(IMAGE_SIZE, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        pixel_values = preprocessor.preprocess(image_base64)

        assert pixel_values.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)

    def test_normalizes_into_the_expected_range(self) -> None:
        import base64
        import io

        from PIL import Image

        image = Image.new("RGB", (32, 32), (255, 255, 255))
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_base64 = base64.b64encode(buf.getvalue()).decode("ascii")

        preprocessor = ClipImagePreprocessor(IMAGE_SIZE, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        pixel_values = preprocessor.preprocess(image_base64)

        # white pixel (1.0) normalized by mean=std=0.5 -> (1.0 - 0.5) / 0.5 = 1.0
        assert torch.allclose(pixel_values, torch.ones_like(pixel_values), atol=1e-5)
