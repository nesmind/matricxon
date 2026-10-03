"""ModelManager hands a retiring model's cached conversations to the disk tier (unload, eviction,
shutdown), and a fresh manager resumes them."""

from pathlib import Path

import pytest
import torch

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
PROMPT = [3 + (i * 5) % 20 for i in range(24)]


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _manager(tmp_path: Path, store: PersistentCacheStore) -> ModelManager:
    return ModelManager(
        ModelCatalog(tmp_path / "models"), max_loaded=1, cache_store=store, prompt_cache_slots=4
    )


def _generate(manager: ModelManager, ids: list[int]) -> int:
    """Runs one reply directly on the handle's engine; returns how many tokens the cache reused."""
    handle = manager.get_or_load(TAG)
    engine = ChatEngine(handle.architecture, {1}, prompt_cache=handle.prompt_cache)
    engine.generate(GenerationRequest(torch.tensor([ids]), SAMPLING))
    return handle.prompt_cache.reused_tokens


@pytest.fixture
def store(tmp_path: Path) -> PersistentCacheStore:
    _install_tiny_model(tmp_path / "models", TAG)
    _install_tiny_model(tmp_path / "models", "model-b:latest")
    policy = PersistencePolicy(True, 100, 24)
    return PersistentCacheStore(tmp_path / "cache", CacheCipher(bytes(range(32))), policy)


@pytest.mark.parametrize("how", ["unload", "shutdown", "evicted_by_another_model"])
def test_a_retired_models_conversation_resumes_on_a_fresh_manager(
    how: str, tmp_path: Path, store: PersistentCacheStore
) -> None:
    first = _manager(tmp_path, store)
    _generate(first, PROMPT)
    if how == "unload":
        first.unload(TAG)
    elif how == "shutdown":
        first.shutdown()
    else:
        first.get_or_load("model-b:latest")  # max_loaded=1: model A is evicted
    store.flush()
    assert store.usage()[0] == 1

    second = _manager(tmp_path, store)
    assert _generate(second, PROMPT + [4, 5, 6]) >= 16


def test_without_a_store_nothing_changes(tmp_path: Path, store: PersistentCacheStore) -> None:
    manager = ModelManager(ModelCatalog(tmp_path / "models"), max_loaded=1)
    _generate(manager, PROMPT)
    manager.shutdown()
    assert store.usage()[0] == 0
