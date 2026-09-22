"""M10: on-the-fly dequant against the real ministral-3:3b GGUF (2GB,

Q4_K/Q6_K) - not just the tiny synthetic fixture tests/unit/test_lazy_
materialization.py uses. Deliberately stops short of ever calling forward()
(full materialization needs ~8GB bf16-estimated RAM - see ModelManager.
_ensure_enough_memory_to_load - more than this machine's ~10GB currently
free at the time this was written): proves construction + defer + close
work against a real large quantized file's real tensor names/shapes/quant
types, without needing to actually pay the full dequant memory cost.
"""

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
        assert model.n_layer == 26  # per ROADMAP.md's real-fixture table

        model.close()
        assert model._pending_loader is None
    finally:
        # No-op if `close()` above already did it - GGUFMemoryMap.close() is idempotent.
        loader.close()
