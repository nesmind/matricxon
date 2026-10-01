"""PromptCache with a hybrid (recurrent) cache: snapshots let a next turn reuse a prefix, and the
result must equal a cold run. Tiny `qwen35` model, greedy, short prompts."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.qwen35 import Qwen35Architecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_qwen35 import EOS_TOKEN_ID, build_tiny_qwen35_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=5, num_ctx=96)
TURN1 = [3 + (i * 7) % 200 for i in range(30)]


@pytest.fixture
def model(tmp_path: Path) -> Iterator[Qwen35Architecture]:
    path = build_tiny_qwen35_gguf(tmp_path / "t.gguf")
    model = Qwen35Architecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    yield model
    model.close()


def _generate(
    model: Qwen35Architecture, cache: PromptCache, ids: list[int], chunk: int = 512
) -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, prefill_chunk=chunk)
    request = GenerationRequest(torch.tensor([ids]), GREEDY)
    return engine.generate(request).token_ids


def test_chunked_prefill_matches_unchunked(model: Qwen35Architecture) -> None:
    assert _generate(model, PromptCache(), TURN1, chunk=7) == _generate(model, PromptCache(), TURN1)


def test_next_turn_restores_the_latest_snapshot_in_the_shared_prefix(
    model: Qwen35Architecture,
) -> None:
    cache = PromptCache()
    _generate(model, cache, TURN1, chunk=8)
    turn2 = TURN1[:25] + [11, 12, 13, 14, 15, 16]

    result = _generate(model, cache, turn2, chunk=8)

    assert cache.reused_tokens == 24  # snapshots at 8/14/16/24/30, newest 4 kept, 24 <= 25
    assert result == _generate(model, PromptCache(), turn2)


def test_a_repeated_prompt_reuses_up_to_its_tail_snapshot(model: Qwen35Architecture) -> None:
    cache = PromptCache()
    first = _generate(model, cache, TURN1)

    assert _generate(model, cache, TURN1) == first
    assert cache.reused_tokens == len(TURN1) - ChatEngine.TAIL_SNAPSHOT_OFFSET


def test_a_divergence_before_every_snapshot_falls_back_to_a_fresh_cache(
    model: Qwen35Architecture,
) -> None:
    cache = PromptCache()
    _generate(model, cache, TURN1, chunk=8)
    turn2 = TURN1[:3] + [99, 98, 97, 96]

    result = _generate(model, cache, turn2, chunk=8)

    assert cache.reused_tokens == 0
    assert result == _generate(model, PromptCache(), turn2)
