"""M10: on-the-fly dequant against the real ministral-3:3b GGUF (2GB, Q4_K/Q6_K), not just the tiny
fixture of tests/unit/test_lazy_materialization.py. Stops short of calling forward (full
materialization needs ~8GB bf16-estimated RAM, ModelManager._ensure_enough_memory_to_load, more than
the ~10GB free here): proves construction + defer + close against a large quantized file's real
tensor names/shapes/quant types without the full dequant memory cost."""

import pytest
import torch

from app.architectures.mistral3 import Mistral3TextArchitecture
from app.gguf.loader import GGUFModelLoader
from tests.integration.conftest import REAL_MINISTRAL_BLOB


def test_from_gguf_defers_materialization_against_the_real_file() -> None:
    if not REAL_MINISTRAL_BLOB.exists():
        pytest.skip(f"real fixture not present on this machine: {REAL_MINISTRAL_BLOB}")

    loader = GGUFModelLoader(REAL_MINISTRAL_BLOB, dtype=torch.bfloat16)
    try:
        model = Mistral3TextArchitecture.from_gguf(loader, dtype=torch.bfloat16)

        assert model.is_materialized is False
        assert model.n_layer == 26

        model.close()
        assert model._pending_loader is None
    finally:
        # No-op if `close` above already did it: GGUFMemoryMap.close is idempotent.
        loader.close()
