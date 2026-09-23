"""PromptCache: a chat's next turn reuses the KV cache for the prefix it shares with the previous
turn (see app/runtime/prompt_cache.py) - and must generate exactly what a fresh cache would."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=6, num_ctx=64)


@pytest.fixture
def model(tmp_path: Path) -> Iterator[LlamaArchitecture]:
    path = build_tiny_llama_gguf(tmp_path / "tiny-llama.gguf")
    model = LlamaArchitecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    yield model
    model.close()


def _request(
    ids: list[int], sampling: SamplingConfig = GREEDY, images: bool = False
) -> GenerationRequest:
    image_embeddings = [(0, torch.zeros(1, 8))] if images else None
    return GenerationRequest(torch.tensor([ids]), sampling, image_embeddings=image_embeddings)


def _generate(
    model: LlamaArchitecture, cache: PromptCache, ids: list[int], **kwargs: object
) -> list[int]:
    engine = ChatEngine(model, eos_token_ids={EOS_TOKEN_ID}, prompt_cache=cache)
    return engine.generate(_request(ids, **kwargs)).token_ids


def test_next_turn_reuses_the_shared_prefix_and_matches_a_fresh_cache(
    model: LlamaArchitecture,
) -> None:
    cache = PromptCache()
    turn1 = [1, 72, 105, 33, 90, 44]
    reply = _generate(model, cache, turn1)
    turn2 = turn1 + reply + [55, 66, 77]

    result = _generate(model, cache, turn2)

    # Every reply token but the last was fed back through the model, so it's in the cache.
    assert cache.reused_tokens == len(turn1) + len(reply) - 1
    assert result == _generate(model, PromptCache(), turn2)


def test_a_repeated_prompt_still_recomputes_its_last_token(model: LlamaArchitecture) -> None:
    cache = PromptCache()
    prompt = [1, 72, 105, 33]
    first = _generate(model, cache, prompt)

    assert _generate(model, cache, prompt) == first
    assert cache.reused_tokens == len(prompt) - 1


def test_a_diverging_prompt_reuses_only_the_common_part(model: LlamaArchitecture) -> None:
    cache = PromptCache()
    _generate(model, cache, [1, 72, 105, 33, 90])

    result = _generate(model, cache, [1, 72, 200, 201])

    assert cache.reused_tokens == 2
    assert result == _generate(model, PromptCache(), [1, 72, 200, 201])


def test_no_reuse_across_a_num_ctx_change(model: LlamaArchitecture) -> None:
    cache = PromptCache()
    _generate(model, cache, [1, 72, 105])

    _generate(model, cache, [1, 72, 105, 33], sampling=SamplingConfig(temperature=0.0, num_ctx=32))

    assert cache.reused_tokens == 0


def test_image_prompts_are_never_reused_or_kept(model: LlamaArchitecture) -> None:
    cache = PromptCache()
    _generate(model, cache, [1, 72, 105])

    _generate(model, cache, [1, 72, 105, 33], images=True)
    assert cache.reused_tokens == 0
    _generate(model, cache, [1, 72, 105, 33, 44])
    assert cache.reused_tokens == 0
