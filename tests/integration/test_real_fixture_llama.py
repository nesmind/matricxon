"""M10: /api/chat against the real TinyLlama-1.1B-Chat-v1.0 GGUF (668MB, Q4_K_M), small enough (bf16
estimate ~2.7GB) for a full real generation, unlike the larger mistral3/gemma4 fixtures. Skips if
not pulled (REAL_LLAMA_TAG). Pulled via matricxon's `/api/pull` into the project's own
`data/models`, a real catalog entry in the configured location (no separate `models_dir`, unlike
REAL_MINISTRAL_BLOB).

Accepted gap this passes despite: `/api/chat` builds its prompt via `Mistral3PromptBuilder` whatever
the architecture (chat_router.py), correct for `mistral3`, wrong for `llama` (TinyLlama's template
uses `<|user|>`/`<|assistant|>`, and `[INST]`/`[/INST]` aren't control tokens in this vocab, just
BPE'd text). Tolerable for a short completion like "The capital of France is", but a multi-turn chat
would get a wrong prompt. Follow-up in ROADMAP.md (per-architecture prompt-builder dispatch), out of
scope for adding llama support.
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

REAL_LLAMA_TAG = "hf.co/hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF:Q4_K_M"
# ~2.7GB bf16 estimate * 1.5x safety margin (ModelManager's threshold): below this the
# InsufficientMemoryError guard would refuse the load anyway, so skip cleanly.
_MIN_AVAILABLE_BYTES = 4 * 1024**3


def _real_llama_installed() -> bool:
    try:
        ModelCatalog(settings.models_dir).get(REAL_LLAMA_TAG)
        return True
    except UnknownModelError:
        return False


def test_real_llama_generates_a_coherent_factual_continuation() -> None:
    """The real-weight bar M4 set for mistral3 ("The capital of France is" -> a coherent, correct
    continuation); greedy decoding is deterministic, so asserting the real-world fact isn't
    flaky."""
    if not _real_llama_installed():
        pytest.skip(f"real fixture not pulled on this machine: {REAL_LLAMA_TAG}")
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
                "model": REAL_LLAMA_TAG,
                "messages": [{"role": "user", "content": "The capital of France is"}],
                "options": {"temperature": 0.0, "num_ctx": 64, "num_predict": 10},
            },
        )
        assert response.status_code == 200
        lines = [json.loads(line) for line in response.text.strip().splitlines()]
        full_text = "".join(line["message"]["content"] for line in lines)
        assert "Paris" in full_text
    finally:
        manager.unload(REAL_LLAMA_TAG)
