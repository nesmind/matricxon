"""A conversation evicted from RAM, or still cached when the server stops, comes back from the
encrypted disk tier - and what it reuses must generate exactly what a cold run would."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.architectures.llama import LlamaArchitecture
from app.architectures.qwen35 import Qwen35Architecture
from app.dependencies import get_cache_store
from app.gguf.loader import GGUFModelLoader
from app.routers import cache_persistence_router
from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=96)
TURN1 = [3 + (i * 7) % 200 for i in range(30)]
OTHER = [5 + (i * 11) % 190 for i in range(30)]


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


@pytest.fixture
def llama(tmp_path: Path) -> Iterator[LlamaArchitecture]:
    model = LlamaArchitecture.from_gguf(
        GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "l.gguf"), dtype=torch.float32)
    )
    yield model
    model.close()


@pytest.fixture
def qwen35(tmp_path: Path) -> Iterator[Qwen35Architecture]:
    model = Qwen35Architecture.from_gguf(
        GGUFModelLoader(build_tiny_qwen35_gguf(tmp_path / "q.gguf"), dtype=torch.float32)
    )
    yield model
    model.close()


def _store(tmp_path: Path) -> PersistentCacheStore:
    policy = PersistencePolicy(True, 100, 24)
    return PersistentCacheStore(tmp_path / "cache", CacheCipher(bytes(range(32))), policy)


def _pool(store: PersistentCacheStore, model_id: str = "m|1", slots: int = 4) -> PromptCache:
    return PromptCache(slots, tier=ModelCacheTier(store, model_id))


def _generate(model: object, cache: PromptCache, ids: list[int], chunk: int = 8) -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=chunk)
    return engine.generate(GenerationRequest(torch.tensor([ids]), GREEDY)).token_ids


@pytest.mark.parametrize("fixture_name", ["llama", "qwen35"])
def test_a_conversation_survives_a_restart(
    fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    model = request.getfixturevalue(fixture_name)
    store = _store(tmp_path)
    before = _pool(store)
    reply = _generate(model, before, TURN1)
    before.persist_all()  # what ModelManager.shutdown does
    store.flush()
    assert store.usage()[0] == 1

    after = _pool(_store(tmp_path))  # a fresh process: new store on the same directory, empty RAM
    turn2 = TURN1 + reply + [55, 66, 77]
    result = _generate(model, after, turn2)

    assert after.reused_tokens >= 16
    assert result == _generate(model, PromptCache(), turn2, chunk=8)


def test_an_evicted_conversation_is_spilled_and_restored(
    llama: LlamaArchitecture, tmp_path: Path
) -> None:
    store = _store(tmp_path)
    pool = _pool(store, slots=1)
    reply = _generate(llama, pool, TURN1)
    _generate(llama, pool, OTHER)  # one slot only: TURN1's conversation is evicted
    store.flush()
    assert store.usage()[0] == 1

    turn2 = TURN1 + reply + [9, 8]
    result = _generate(llama, pool, turn2)

    assert pool.reused_tokens >= 24
    assert result == _generate(llama, PromptCache(), turn2)
    assert store.usage()[0] == 1  # the restore consumed TURN1's file; OTHER got evicted in turn


def test_another_model_never_restores_it(llama: LlamaArchitecture, tmp_path: Path) -> None:
    store = _store(tmp_path)
    pool = _pool(store, "model-a|1")
    reply = _generate(llama, pool, TURN1)
    pool.persist_all()
    store.flush()

    other = _pool(store, "model-b|1")
    _generate(llama, other, TURN1 + reply + [1, 2])
    assert other.reused_tokens == 0
    assert store.usage()[0] == 1


def test_a_disabled_tier_changes_nothing(llama: LlamaArchitecture, tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.policy.enabled = False
    pool = _pool(store)
    _generate(llama, pool, TURN1)
    pool.persist_all()
    store.flush()
    assert store.usage()[0] == 0


def test_admin_endpoint_reads_updates_and_clears(llama: LlamaArchitecture, tmp_path: Path) -> None:
    store = _store(tmp_path)
    pool = _pool(store)
    _generate(llama, pool, TURN1)
    pool.persist_all()
    store.flush()
    app = FastAPI()
    app.include_router(cache_persistence_router.router)
    app.dependency_overrides[get_cache_store] = lambda: store
    client = TestClient(app)

    status = client.get("/api/cache/persistence").json()
    assert status["enabled"] and status["files"] == 1 and status["used_bytes"] > 0

    changed = client.put("/api/cache/persistence", json={"budget_mb": 7, "ttl_hours": 2}).json()
    assert (changed["budget_mb"], changed["ttl_hours"], changed["files"]) == (7, 2, 1)

    assert client.put("/api/cache/persistence", json={"ttl_hours": 0}).status_code == 422

    off = client.put("/api/cache/persistence", json={"enabled": False}).json()
    assert off["enabled"] is False and off["files"] == 0  # disabling also wipes the files
