"""Context shifting: when a chat's oldest messages are trimmed, the cache drops those tokens and
keeps the rest (keys re-rotated to their new positions) instead of re-reading the whole chat."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.rope import apply_rotary_pos_emb
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.kv_cache import KVCache
from app.runtime.prompt_cache import PromptCache
from app.runtime.prompt_slot import CacheSlot
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_gguf
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

SYSTEM = [3 + (i * 5) % 90 for i in range(12)]
OLD = [100 + (i * 7) % 90 for i in range(20)]  # the oldest messages, trimmed away later
MID = [20 + (i * 11) % 80 for i in range(24)]
TAIL1 = [150, 151, 152, 153]
TAIL2 = [160, 161, 162, 163, 164]
GREEDY = SamplingConfig(temperature=0.0, num_predict=2, num_ctx=160)


@pytest.fixture(autouse=True)
def short_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CacheSlot, "MIN_RUN_AFTER_GAP", 8)


@pytest.fixture(params=["llama", "gemma4"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[object]:
    if request.param == "llama":
        arch = LlamaArchitecture.from_gguf(
            GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "m.gguf"), dtype=torch.float32)
        )
    else:
        arch = Gemma4Architecture.from_gguf(
            GGUFModelLoader(build_tiny_gemma4_gguf(tmp_path / "m.gguf"), dtype=torch.float32),
            dtype=torch.float32,
        )
    yield arch
    arch.close()


def _run(model: object, cache: PromptCache, ids: list[int], tag: str = "chat") -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=8)
    request = GenerationRequest(torch.tensor([ids]), GREEDY, cache_tag=tag)
    return engine.generate(request).token_ids


def _slot_of(cache: PromptCache) -> CacheSlot:
    return cache._slots[0]


def test_the_gap_is_found_where_the_oldest_messages_were_trimmed():
    slot = CacheSlot(KVCache([(1, 4)], 200), (200, torch.float32), SYSTEM + OLD + MID + TAIL1)
    new = SYSTEM + MID + TAIL1 + [9, 9]
    assert slot.find_gap(new) == (len(SYSTEM), len(OLD), len(MID) + len(TAIL1))


def test_no_gap_for_a_plain_continuation_or_an_unrelated_prompt():
    slot = CacheSlot(KVCache([(1, 4)], 200), (200, torch.float32), SYSTEM + OLD + MID)
    assert slot.find_gap(SYSTEM + OLD + MID + [1, 2, 3]) is None  # nothing was cut
    assert slot.find_gap(SYSTEM + [250] * 40) is None  # nothing lines up again
    assert slot.find_gap([250] * 40) is None  # no shared start at all


def test_shifted_keys_equal_keys_rotated_fresh_for_their_new_positions(model):
    """Every layer type (Gemma 4's local and global ropes have different frequencies and widths)."""
    n, start, count = 40, 5, 12
    shapes = model.kv_cache_layer_shapes
    cache = KVCache(shapes, 64)
    raw = [torch.randn(1, heads, n, dim) for heads, dim in shapes]
    for i in range(len(shapes)):
        cos, sin = model._rope_for_layer(i)(torch.arange(n))
        cache._k[i][:, :, :n] = apply_rotary_pos_emb(raw[i], raw[i], cos, sin)[0]
        cache._v[i][:, :, :n] = raw[i]
    cache._length = n

    cache.drop_range(start, count, model.context_shift_rotations(count))

    assert cache.length == n - count
    for i in range(len(shapes)):
        kept = torch.cat([raw[i][:, :, :start], raw[i][:, :, start + count :]], dim=2)
        cos, sin = model._rope_for_layer(i)(torch.arange(n - count))
        fresh = apply_rotary_pos_emb(kept, kept, cos, sin)[0]
        assert torch.allclose(cache._k[i][:, :, : n - count], fresh, atol=1e-4)
        assert torch.equal(cache._v[i][:, :, : n - count], kept)


def test_a_trimmed_chat_reuses_its_cache_and_the_first_layer_matches_a_cold_read(model, caplog):
    cache = PromptCache(4)
    _run(model, cache, SYSTEM + OLD + MID + TAIL1)
    trimmed = SYSTEM + MID + TAIL1 + [9, 8, 7] + TAIL2

    with caplog.at_level(logging.INFO):
        warm = _run(model, cache, trimmed)

    assert "context shift: dropped" in caplog.text
    assert cache.reused_tokens >= len(SYSTEM) + len(MID)  # everything after the cut was kept
    assert len(warm) > 0
    cold = PromptCache(4)
    _run(model, cold, trimmed)
    n = len(trimmed)
    # layer 0's keys depend only on the token and its position, so they are exact after the shift
    assert torch.allclose(
        _slot_of(cache).cache._k[0][:, :, :n], _slot_of(cold).cache._k[0][:, :, :n], atol=1e-4
    )


def test_switching_it_off_reads_the_whole_chat_again(model, caplog):
    cache = PromptCache(4, context_shift=False)
    _run(model, cache, SYSTEM + OLD + MID + TAIL1)

    with caplog.at_level(logging.INFO):
        _run(model, cache, SYSTEM + MID + TAIL1 + [9, 8, 7])

    assert "context shift" not in caplog.text
    assert cache.reused_tokens <= len(SYSTEM)  # only the shared start survives


def test_another_chats_cache_is_never_cut(model, caplog):
    cache = PromptCache(4)
    _run(model, cache, SYSTEM + OLD + MID + TAIL1, tag="chat-a")

    with caplog.at_level(logging.INFO):
        _run(model, cache, SYSTEM + MID + TAIL1 + [9, 8, 7], tag="chat-b")

    assert "context shift" not in caplog.text
