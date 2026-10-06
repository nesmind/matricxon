"""`num_ctx` is only a cache's capacity: a different one must not cost a chat its cached prefix, and
what is reused must generate exactly what a cold run does."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.architectures.qwen35 import Qwen35Architecture
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
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

TURN1 = [3 + (i * 7) % 200 for i in range(30)]
TURN2 = TURN1 + [9, 10, 11, 12]


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


@pytest.fixture(params=["llama", "qwen35"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[object]:
    if request.param == "llama":
        arch = LlamaArchitecture.from_gguf(
            GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "m.gguf"), dtype=torch.float32)
        )
    else:  # a hybrid cache: reuse goes through recurrent snapshots
        arch = Qwen35Architecture.from_gguf(
            GGUFModelLoader(build_tiny_qwen35_gguf(tmp_path / "m.gguf"), dtype=torch.float32)
        )
    yield arch
    arch.close()


def _run(model: object, cache: PromptCache, ids: list[int], num_ctx: int) -> list[int]:
    sampling = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=num_ctx)
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=8)
    return engine.generate(
        GenerationRequest(torch.tensor([ids]), sampling, cache_tag="c")
    ).token_ids


@pytest.mark.parametrize(("first", "second"), [(96, 160), (160, 96)])
def test_a_changed_num_ctx_still_reuses_the_prefix_and_generates_the_same(
    model, first, second, caplog
):
    cold = _run(model, PromptCache(4), TURN2, second)
    cache = PromptCache(4)
    _run(model, cache, TURN1, first)

    with caplog.at_level(logging.INFO):
        warm = _run(model, cache, TURN2, second)

    assert warm == cold
    assert cache.reused_tokens > 0 and "re-reading the entire chat" not in caplog.text
    assert max(s.key[0] for s in cache._slots) >= second


def test_growing_replaces_the_smaller_slot_of_the_same_chat(model):
    cache = PromptCache(4)
    _run(model, cache, TURN1, 96)
    _run(model, cache, TURN2, 160)
    assert [s.key[0] for s in cache._slots] == [160]


def test_a_disk_file_saved_with_another_num_ctx_is_restored(model, tmp_path):
    policy = PersistencePolicy(True, 100, 24)
    store = PersistentCacheStore(tmp_path / "cache", CacheCipher(bytes(range(32))), policy)
    before = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
    _run(model, before, TURN1, 96)
    before.persist_all()
    store.flush()
    cold = _run(model, PromptCache(4), TURN2, 160)

    after = PromptCache(4, tier=ModelCacheTier(store, "m|1"))  # a restarted server
    warm = _run(model, after, TURN2, 160)

    assert warm == cold and after.reused_tokens > 0


def test_a_reply_stops_at_the_end_of_the_context_instead_of_failing(model):
    """num_predict larger than the room left (long chat, long reply): the reply is cut at num_ctx
    finishes as "length" - it used to raise PromptTooLongError halfway through the stream."""
    ids = TURN1  # 30 tokens
    sampling = SamplingConfig(temperature=0.0, num_predict=500, num_ctx=len(ids) + 6)
    engine = ChatEngine(model, set(), prompt_cache=PromptCache(4), prefill_chunk=8)

    result = engine.generate(GenerationRequest(torch.tensor([ids]), sampling, cache_tag="c"))

    assert len(result.token_ids) == 6 and result.finish_reason == "length"
