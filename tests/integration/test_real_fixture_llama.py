"""M10: /api/chat against the real TinyLlama-1.1B-Chat-v1.0 GGUF (668MB,

Q4_K_M) - small enough (bf16 estimate ~2.7GB) to run a full real generation
safely, unlike the much larger real mistral3/gemma4 fixtures. Skips
automatically if this machine hasn't pulled it (see REAL_LLAMA_TAG below) -
pulled via matricxon's own real `/api/pull` into this project's own
`data/models`, not an external blob store, so (unlike REAL_MINISTRAL_BLOB)
there's no separate `models_dir` to point at: it's already a real catalog
entry in the app's own configured location.

Known, accepted gap this test still passes despite: `/api/chat` always
builds its prompt via `Mistral3PromptBuilder` regardless of which
architecture is actually loaded (see chat_router.py) - correct for
`mistral3`, structurally wrong for `llama` (TinyLlama's real chat template
uses `<|user|>`/`<|assistant|>`, not `[INST]`/`[/INST]`, which this vocab
doesn't even have as control tokens - they just become ordinary BPE'd
text). Tolerable for a short factual completion like the one below (the
literal "[INST]"/"[/INST]" text doesn't stop the model from still finding
and continuing "The capital of France is" correctly), but a real multi-turn
chat conversation would get a genuinely wrong prompt structure. Flagged in
ROADMAP.md as real follow-up work (per-architecture prompt-builder
dispatch), not fixed here - out of scope for "add llama architecture
support" itself.
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
# ~2.7GB bf16 estimate * 1.5x safety margin (ModelManager's own threshold) -
# below this, the real InsufficientMemoryError guard would legitimately
# refuse the load anyway, so skip cleanly instead of failing noisily.
_MIN_AVAILABLE_BYTES = 4 * 1024**3


def _real_llama_installed() -> bool:
    try:
        ModelCatalog(settings.models_dir).get(REAL_LLAMA_TAG)
        return True
    except UnknownModelError:
        return False


def test_real_llama_generates_a_coherent_factual_continuation() -> None:
    """Same real-weight validation bar M4 set for mistral3 ("The capital of

    France is" -> a coherent, factually correct continuation) - greedy
    decoding is fully deterministic, so asserting on the actual real-world
    fact is safe, not flaky.
    """
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
