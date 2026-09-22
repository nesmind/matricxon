"""`Settings.log_level` (0-2, see its own docstring) and the level-2 per-decoder-layer trace it
gates - app.main.MatricxonApp maps this onto Python's stdlib `logging` levels (0->WARNING,
1->INFO, 2->DEBUG); this file checks the setting's own bounds and, separately, that the level-2
trace this session added to Mistral3TextArchitecture._forward_impl actually fires once per real
layer on a real (if tiny) forward pass - not just that the log call exists in the source.
"""

import logging
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.config import Settings
from app.gguf.loader import GGUFModelLoader
from tests.tiny_gguf import build_tiny_mistral3_gguf


class TestLogLevelBounds:
    def test_defaults_to_0(self) -> None:
        # _env_file=None: isolates this specific check from a real .env file that may genuinely
        # exist in the repo root (see Settings.model_config's own docstring for why one now does,
        # by design) - this test is about the hardcoded fallback, not this machine's local config.
        assert Settings(_env_file=None).log_level == 0

    @pytest.mark.parametrize("level", [0, 1, 2])
    def test_accepts_0_through_2(self, level: int) -> None:
        assert Settings(log_level=level).log_level == level

    @pytest.mark.parametrize("level", [-1, 3])
    def test_rejects_out_of_range(self, level: int) -> None:
        with pytest.raises(ValidationError):
            Settings(log_level=level)


class TestPerLayerTrace:
    def test_debug_level_emits_one_line_per_layer(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        n_layer = 3
        gguf_path = tmp_path / "tiny.gguf"
        build_tiny_mistral3_gguf(gguf_path, n_layer=n_layer)
        loader = GGUFModelLoader(gguf_path, dtype=torch.float32)
        model = Mistral3TextArchitecture.from_gguf(loader, dtype=torch.float32)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        with caplog.at_level(logging.DEBUG, logger="app.architectures.mistral3"):
            model(input_ids)

        layer_lines = [r for r in caplog.records if r.message.startswith("layer ")]
        assert len(layer_lines) == n_layer
        assert [r.message.split(":")[0] for r in layer_lines] == [
            f"layer {i + 1}/{n_layer}" for i in range(n_layer)
        ]

    def test_info_level_emits_no_per_layer_lines(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        gguf_path = tmp_path / "tiny.gguf"
        build_tiny_mistral3_gguf(gguf_path, n_layer=3)
        loader = GGUFModelLoader(gguf_path, dtype=torch.float32)
        model = Mistral3TextArchitecture.from_gguf(loader, dtype=torch.float32)
        model.eval()
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        with caplog.at_level(logging.INFO, logger="app.architectures.mistral3"):
            model(input_ids)

        assert not any(r.message.startswith("layer ") for r in caplog.records)
