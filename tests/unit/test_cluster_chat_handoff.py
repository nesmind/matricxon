"""Moving a chat between nodes: a node reports whether it holds a chat, spills just that chat to the
shared disk
(the model stays loaded) and another node resumes it.
"""

import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from app.dependencies import get_model_manager
from app.main import MatricxonApp
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from tests.unit.test_model_manager import _install_tiny_model

TAG = "model-a:latest"
SAMPLING = SamplingConfig(temperature=0.0, num_predict=3, num_ctx=64)
PROMPTS = {
    "chat-1": [3 + (i * 5) % 20 for i in range(24)],
    "chat-2": [4 + (i * 7) % 19 for i in range(24)],
    "busy": [5 + (i * 3) % 18 for i in range(24)],
}
KEY = bytes(range(32))


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _node(tmp_path: Path, name: str) -> ModelManager:
    """A node: its own manager and store object, one cache directory shared with the other nodes."""
    store = PersistentCacheStore(
        tmp_path / "shared-cache", CacheCipher(KEY), PersistencePolicy(True, 100, 24), node_id=name
    )
    return ModelManager(
        ModelCatalog(tmp_path / "models"), max_loaded=1, cache_store=store, prompt_cache_slots=4
    )


def _generate(manager: ModelManager, chat: str) -> int:
    handle = manager.get_or_load(TAG)
    engine = ChatEngine(handle.architecture, {1}, prompt_cache=handle.prompt_cache)
    request = GenerationRequest(torch.tensor([PROMPTS[chat]]), SAMPLING, cache_tag=chat)
    engine.generate(request)
    return handle.prompt_cache.reused_tokens


@pytest.fixture
def models(tmp_path: Path) -> None:
    _install_tiny_model(tmp_path / "models", TAG)


def test_a_node_reports_the_chats_it_holds(tmp_path: Path, models: None) -> None:
    node = _node(tmp_path, "a")
    assert not node.holds_chat("chat-1")
    _generate(node, "chat-1")
    assert node.holds_chat("chat-1") and not node.holds_chat("chat-2") and not node.holds_chat("")


def test_spilling_one_chat_keeps_the_model_loaded_and_lets_another_node_resume_it(
    tmp_path: Path, models: None
) -> None:
    a, b = _node(tmp_path, "a"), _node(tmp_path, "b")
    _generate(a, "chat-1")
    _generate(a, "chat-2")

    assert a.spill_chat("chat-1") == 1
    assert not a.holds_chat("chat-1") and a.holds_chat("chat-2")
    assert [h.tag for h in a.list_loaded()] == [TAG]
    assert len(list((tmp_path / "shared-cache").glob("*.mxc"))) == 1

    time.sleep(0.15)
    assert _generate(b, "chat-1") > 0
    assert not list((tmp_path / "shared-cache").glob("*.mxc"))


def test_spilling_an_unknown_chat_does_nothing(tmp_path: Path, models: None) -> None:
    node = _node(tmp_path, "a")
    _generate(node, "chat-1")
    assert node.spill_chat("nope") == 0 and node.spill_chat("") == 0 and node.holds_chat("chat-1")


def test_a_chat_still_replying_is_not_spilled(tmp_path: Path, models: None) -> None:
    node = _node(tmp_path, "a")
    handle = node.get_or_load(TAG)
    pool = handle.prompt_cache
    cache, _ = pool.acquire(handle.architecture, PROMPTS["busy"], 64, torch.float32, True, "busy")
    assert node.spill_chat("busy") == 0 and node.holds_chat("busy")
    pool.release(cache)
    assert node.spill_chat("busy") == 1


def test_a_lookup_that_misses_rescans_so_a_just_written_handoff_is_found(tmp_path: Path) -> None:
    reader = PersistentCacheStore(
        tmp_path / "c",
        CacheCipher(KEY),
        PersistencePolicy(True, 100, 24),
        "r",
        rescan_seconds=3600.0,
    )
    writer = PersistentCacheStore(
        tmp_path / "c", CacheCipher(KEY), PersistencePolicy(True, 100, 24), "w"
    )
    reader.MISS_RESCAN_SECONDS = 0.0
    from tests.unit.test_cache_store import _slot

    writer.spill_async("m|1", _slot(list(range(24))))
    writer.flush()
    # the reader never rescanned on its own (3600 s), yet the miss triggers one
    assert reader.take("m|1", list(range(24)) + [1], torch.float32, 0) is not None


def test_endpoints_report_and_spill(tmp_path: Path, models: None) -> None:
    node = _node(tmp_path, "a")
    _generate(node, "chat-1")
    app = MatricxonApp().build()
    app.dependency_overrides[get_model_manager] = lambda: node
    client = TestClient(app)
    assert client.get("/api/cache/holds", params={"tag": "chat-1"}).json() == {"ram": True}
    assert client.get("/api/cache/holds", params={"tag": "x"}).json() == {"ram": False}
    assert client.post("/api/cache/spill", json={"tag": "chat-1"}).json() == {"spilled": 1}
    assert client.get("/api/cache/holds", params={"tag": "chat-1"}).json() == {"ram": False}
