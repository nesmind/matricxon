"""M10: on-the-fly dequant - `ModelArchitecture` defers copying real weight

data in until its first real forward pass, caching the result after that
(see app/architectures/base.py's own docstring). Exercised here against the
real `Mistral3TextArchitecture`/tiny GGUF pipeline, not a fake stand-in -
this is precisely the mechanism that matters, not just wiring.
"""

from pathlib import Path

import pytest
import torch

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf import build_tiny_mistral3_gguf


def _load(tmp_path: Path) -> tuple[Mistral3TextArchitecture, GGUFModelLoader]:
    gguf_path = tmp_path / "tiny.gguf"
    build_tiny_mistral3_gguf(gguf_path)
    loader = GGUFModelLoader(gguf_path, dtype=torch.float32)
    model = Mistral3TextArchitecture.from_gguf(loader, dtype=torch.float32)
    model.eval()
    return model, loader


class TestDeferredMaterialization:
    def test_from_gguf_does_not_materialize_weights(self, tmp_path: Path) -> None:
        model, _loader = _load(tmp_path)

        assert model.is_materialized is False

    def test_first_forward_pass_materializes_weights(self, tmp_path: Path) -> None:
        model, _loader = _load(tmp_path)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        model(input_ids)

        assert model.is_materialized is True

    def test_first_forward_pass_closes_the_loader(self, tmp_path: Path) -> None:
        model, loader = _load(tmp_path)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        model(input_ids)

        with pytest.raises(Exception):  # noqa: B017, PT011 - mmap access after close, any error
            loader.load_tensor("token_embd.weight")

    def test_second_forward_pass_reuses_cached_weights_without_touching_loader(
        self, tmp_path: Path
    ) -> None:
        model, _loader = _load(tmp_path)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        first = model(input_ids)
        second = model(input_ids)  # would crash if this tried to re-read the (closed) loader

        assert torch.equal(first, second)

    def test_close_before_any_forward_pass_releases_the_loader_without_materializing(
        self, tmp_path: Path
    ) -> None:
        model, _loader = _load(tmp_path)

        model.close()

        assert model.is_materialized is False
        assert model._pending_loader is None

    def test_close_after_materialization_is_a_safe_no_op(self, tmp_path: Path) -> None:
        model, _loader = _load(tmp_path)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        model(input_ids)

        model.close()  # must not raise

        assert model.is_materialized is True

    def test_forward_after_close_raises_a_clear_error(self, tmp_path: Path) -> None:
        model, _loader = _load(tmp_path)
        model.close()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        with pytest.raises(RuntimeError, match="evicted/closed"):
            model(input_ids)
