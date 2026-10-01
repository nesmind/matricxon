"""The pool keeps one cache per recent conversation, so users sharing a model don't wipe each
other's cache - and whatever it reuses must generate exactly what a cold run would."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.architectures.qwen35 import Qwen35Architecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=96)
CONV_A = [1, 72, 105, 33, 90, 44]
CONV_B = [1, 200, 201, 202, 203, 204]


@pytest.fixture
def llama(tmp_path: Path) -> Iterator[LlamaArchitecture]:
    path = build_tiny_llama_gguf(tmp_path / "llama.gguf")
    model = LlamaArchitecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    yield model
    model.close()


@pytest.fixture
def qwen35(tmp_path: Path) -> Iterator[Qwen35Architecture]:
    path = build_tiny_qwen35_gguf(tmp_path / "qwen35.gguf")
    model = Qwen35Architecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    yield model
    model.close()


def _pool(**kwargs: int) -> PromptCache:
    """Tiny prompts drop fewer tokens than the real fork threshold - lower it to exercise forks."""
    cache = PromptCache(**kwargs)
    cache.FORK_MIN_DROPPED_TOKENS = 1
    return cache


def _generate(model: object, cache: PromptCache, ids: list[int], chunk: int = 512) -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=chunk)
    return engine.generate(GenerationRequest(torch.tensor([ids]), GREEDY)).token_ids


def _two_users_alternate(model: object, cache: PromptCache, chunk: int = 512) -> None:
    reply_a = _generate(model, cache, CONV_A, chunk)
    reply_b = _generate(model, cache, CONV_B, chunk)
    turn_a2, turn_b2 = CONV_A + reply_a + [55, 66], CONV_B + reply_b + [77, 88]

    out_a2 = _generate(model, cache, turn_a2, chunk)
    assert cache.reused_tokens > 0  # A's cache survived B's request
    out_b2 = _generate(model, cache, turn_b2, chunk)
    assert cache.reused_tokens > 0

    assert out_a2 == _generate(model, PromptCache(), turn_a2, chunk)
    assert out_b2 == _generate(model, PromptCache(), turn_b2, chunk)


def test_two_users_alternating_each_keep_their_own_cache(llama: LlamaArchitecture) -> None:
    cache = _pool()
    _two_users_alternate(llama, cache)
    assert cache.slot_count == 2


def test_hybrid_model_users_alternating_keep_their_own_snapshots(
    qwen35: Qwen35Architecture,
) -> None:
    cache = _pool()
    _two_users_alternate(qwen35, cache, chunk=3)
    assert cache.slot_count == 2


def test_a_diverging_prompt_forks_and_leaves_the_original_conversation_intact(
    llama: LlamaArchitecture,
) -> None:
    cache = _pool()
    reply = _generate(llama, cache, CONV_A)
    _generate(llama, cache, CONV_A[:3] + [150, 151, 152])  # shares 3 tokens, then differs

    assert cache.slot_count == 2
    assert cache.reused_tokens == 3
    follow_up = CONV_A + reply + [55]
    out = _generate(llama, cache, follow_up)
    assert cache.reused_tokens == len(CONV_A) + len(reply) - 1  # the original was not cut back
    assert out == _generate(llama, PromptCache(), follow_up)


def test_with_a_single_slot_users_take_turns_correctly_without_reuse(
    llama: LlamaArchitecture,
) -> None:
    cache = _pool(max_slots=1)
    reply_a = _generate(llama, cache, CONV_A)
    _generate(llama, cache, CONV_B)
    turn_a2 = CONV_A + reply_a + [55]

    out = _generate(llama, cache, turn_a2)

    assert cache.slot_count == 1
    assert cache.reused_tokens == 1  # only the shared first token: B replaced A's cache
    assert out == _generate(llama, PromptCache(), turn_a2)


def test_a_tiny_memory_budget_still_keeps_one_slot(llama: LlamaArchitecture) -> None:
    cache = _pool(max_slots=8, budget_bytes=1)
    reply_a = _generate(llama, cache, CONV_A)
    _generate(llama, cache, CONV_B)
    assert cache.slot_count == 1

    turn_b2 = CONV_B + [1, 2]
    assert _generate(llama, cache, turn_b2) == _generate(llama, PromptCache(), turn_b2)
    assert reply_a  # generation itself was unaffected


def test_slots_are_evicted_least_recently_used_first(llama: LlamaArchitecture) -> None:
    cache = _pool(max_slots=2)
    reply_a = _generate(llama, cache, CONV_A)
    _generate(llama, cache, CONV_B)
    _generate(llama, cache, CONV_A + reply_a + [9])  # touches A -> B is now the oldest
    _generate(llama, cache, [1, 250, 251, 252, 253])  # a third conversation evicts B

    assert cache.slot_count == 2
    _generate(llama, cache, CONV_A + reply_a + [9, 8])
    assert cache.reused_tokens > 0  # A still cached


def test_a_continuing_conversation_stays_in_one_slot(llama: LlamaArchitecture) -> None:
    cache = _pool()
    prompt = list(CONV_A)
    for new_message in ([55, 66], [77, 88], [99, 100]):
        reply = _generate(llama, cache, prompt)
        prompt = prompt + reply + new_message

    assert cache.slot_count == 1  # no stale copies piling up as the chat goes on
