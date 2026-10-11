"""Real-weight validation for phi2 (moondream2's text model): the bar M4/M10 set for mistral3/llama
("The capital of France is" -> a coherent, correct continuation), against the moondream2 GGUF pulled
via matricxon's `/api/pull` (metadata/tensor findings: app/architectures/phi2.py). Skips if not
pulled.

This resolves the open design question in app/architectures/phi2_layers.py (whether the fused
attn_qkv's q/k thirds need the row permutation `unpermute_rope_rows` applies to other rotate-half
architectures): if this fails with incoherent output instead of an error, try that permutation first
in Phi2Architecture._materialize_weights.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_model_catalog, get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.load_dtype import available_memory_bytes
from app.models.manager import ModelManager
from app.server.errors import UnknownModelError

REAL_MOONDREAM2_TAG = "hf.co/moondream/moondream2-gguf:text"
# moondream2's text model is ~2.84GB f16 on disk -> ~3.1GB bf16 estimate * 1.5x safety margin:
# small, but skip cleanly under memory pressure rather than fail noisily (as
# test_real_fixture_llama.py).
_MIN_AVAILABLE_BYTES = 4 * 1024**3


def _real_moondream2_installed() -> bool:
    try:
        ModelCatalog(settings.models_dir).get(REAL_MOONDREAM2_TAG)
        return True
    except UnknownModelError:
        return False


def test_real_moondream2_generates_a_coherent_factual_continuation() -> None:
    if not _real_moondream2_installed():
        pytest.skip(f"real fixture not pulled on this machine: {REAL_MOONDREAM2_TAG}")
    available = available_memory_bytes()
    if available is not None and available < _MIN_AVAILABLE_BYTES:
        pytest.skip(
            f"not enough free memory to safely load a real model right now "
            f"({available / 1e9:.1f}GB available)"
        )

    catalog = ModelCatalog(settings.models_dir)
    manager = ModelManager(catalog)
    app = MatricxonApp().build()
    app.dependency_overrides[get_model_catalog] = lambda: catalog
    app.dependency_overrides[get_model_manager] = lambda: manager
    client = TestClient(app)

    try:
        response = client.post(
            "/api/chat",
            json={
                "model": REAL_MOONDREAM2_TAG,
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "options": {"temperature": 0.0, "num_ctx": 64, "num_predict": 10},
            },
        )
        assert response.status_code == 200
        lines = [json.loads(line) for line in response.text.strip().splitlines()]
        full_text = "".join(line["message"]["content"] for line in lines)
        assert "Paris" in full_text
    finally:
        manager.unload(REAL_MOONDREAM2_TAG)
