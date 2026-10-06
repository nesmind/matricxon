"""A prompt that reuses nothing logs "re-reading the entire chat" with the reason."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

TURN1 = [3 + (i * 7) % 200 for i in range(30)]
OTHER = [5 + (i * 11) % 190 for i in range(30)]
MARK = "re-reading the entire chat"


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


def _store(tmp_path: Path) -> PersistentCacheStore:
    policy = PersistencePolicy(True, 100, 24)
    return PersistentCacheStore(tmp_path / "cache", CacheCipher(bytes(range(32))), policy)


def _run(
    model: object, cache: PromptCache, ids: list[int], num_ctx: int = 96, tag: str = "c"
) -> None:
    sampling = SamplingConfig(temperature=0.0, num_predict=2, num_ctx=num_ctx)
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=8)
    engine.generate(GenerationRequest(torch.tensor([ids]), sampling, cache_tag=tag))


def _notes(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if MARK in r.getMessage()]


def test_a_cold_start_logs_why_nothing_was_reused(llama, tmp_path, caplog):
    cache = PromptCache(4, tier=ModelCacheTier(_store(tmp_path), "m|1"))
    with caplog.at_level(logging.INFO):
        _run(llama, cache, TURN1)
    [note] = _notes(caplog)
    assert "30 tokens" in note and "RAM holds no conversations" in note
    assert "disk cache is empty" in note


def test_a_reused_prefix_logs_nothing(llama, caplog):
    cache = PromptCache(4)
    _run(llama, cache, TURN1)
    with caplog.at_level(logging.INFO):
        _run(llama, cache, TURN1 + [9, 10, 11])
    assert _notes(caplog) == []


def test_a_different_dtype_is_named_as_the_reason(llama, caplog):
    cache = PromptCache(4)
    _run(llama, cache, TURN1)
    slot_dtype = cache._slots[0].key[1]
    cache._slots[0].key = (96, torch.float16 if slot_dtype != torch.float16 else torch.float32)
    with caplog.at_level(logging.INFO):
        _run(llama, cache, TURN1 + [9, 10, 11])
    [note] = _notes(caplog)
    assert "this chat's RAM cache uses" in note


def test_a_file_from_another_chat_is_reported(llama, tmp_path, caplog):
    store = _store(tmp_path)
    before = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    _run(llama, before, TURN1, tag="other")
    before.persist_all()
    store.flush()

    after = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    with caplog.at_level(logging.INFO):
        _run(llama, after, OTHER, tag="mine")
    [note] = _notes(caplog)
    assert "no file for this chat on disk (1 from other chats)" in note
