"""A plain KV cache's prefill is chunked, so a cancel keeps the pieces already committed."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.qwen35 import Qwen35Architecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_gguf
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

GREEDY = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=96)
PROMPT = [1, 72, 105, 33, 90, 44, 12, 13, 14, 15, 16, 17, 18, 19, 20]


@pytest.fixture
def llama(tmp_path: Path) -> Iterator[LlamaArchitecture]:
    path = build_tiny_llama_gguf(tmp_path / "llama.gguf")
    model = LlamaArchitecture.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
    yield model
    model.close()


def _stop_after(checks: int) -> Callable[[], bool]:
    """A stop_check that starts reporting a stop after `checks` calls (one per layer)."""
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > checks

    return stop


def _engine(model: object, cache: PromptCache, chunk: int) -> ChatEngine:
    return ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=cache, plain_prefill_chunk=chunk)


def test_cuts_split_plain_prompt(llama: LlamaArchitecture) -> None:
    engine = _engine(llama, PromptCache(), 4)
    assert engine._prefill_cuts(object(), 2, 15) == [6, 10, 14, 15]


def test_chunked_prefill_matches_whole_prompt(llama: LlamaArchitecture) -> None:
    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    whole = _engine(llama, PromptCache(), 512).generate(request).token_ids
    assert _engine(llama, PromptCache(), 4).generate(request).token_ids == whole


def test_cancelled_prefill_keeps_committed_pieces(llama: LlamaArchitecture) -> None:
    cache = PromptCache()
    calls = {"n": 0}

    def stop_on_third_piece() -> bool:
        calls["n"] += 1
        return calls["n"] > 2 * llama.n_layer

    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    engine = _engine(llama, cache, 4)
    assert list(engine.stream(request, stop_on_third_piece)) == []  # cancelled in prefill
    engine.generate(request)
    assert cache.reused_tokens >= 4  # at least the first piece


@pytest.mark.parametrize("chunk", [1, 3, 4, 7])
def test_chunked_prefill_is_exact_on_other_architectures(tmp_path: Path, chunk: int) -> None:
    """Gemma4 (sliding-window + global layers) and Qwen3.5 (hybrid) must not drift when split."""
    builds = [
        (Gemma4Architecture, build_tiny_gemma4_gguf(tmp_path / "g.gguf")),
        (Qwen35Architecture, build_tiny_qwen35_gguf(tmp_path / "q.gguf")),
    ]
    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    for arch, path in builds:
        model = arch.from_gguf(GGUFModelLoader(path, dtype=torch.float32))
        try:
            whole = _engine(model, PromptCache(), 512).generate(request).token_ids
            split = ChatEngine(
                model,
                {EOS_TOKEN_ID},
                prompt_cache=PromptCache(),
                prefill_chunk=chunk,
                plain_prefill_chunk=chunk,
            )
            assert split.generate(request).token_ids == whole
        finally:
            model.close()


def test_image_prompt_is_never_split(llama: LlamaArchitecture) -> None:
    engine = _engine(llama, PromptCache(), 4)
    assert engine._prefill_cuts(object(), 0, 15, has_images=True) == [15]


@pytest.mark.parametrize(
    ("reused", "prompt_len", "expected"),
    [
        (0, 3, [3]),  # shorter than a piece
        (0, 4, [4]),  # exactly one piece
        (0, 8, [4, 8]),  # exact multiple: no empty last piece
        (0, 9, [4, 8, 9]),
        (5, 9, [9]),  # remainder smaller than a piece
        (5, 14, [9, 13, 14]),  # pieces start at the reused prefix
    ],
)
def test_cuts_edge_cases(llama: LlamaArchitecture, reused: int, prompt_len: int, expected) -> None:
    assert _engine(llama, PromptCache(), 4)._prefill_cuts(object(), reused, prompt_len) == expected


def test_cuts_never_empty_or_past_prompt(llama: LlamaArchitecture) -> None:
    engine = _engine(llama, PromptCache(), 5)
    for reused in range(0, 20):
        for prompt_len in range(reused + 1, 40):
            cuts = engine._prefill_cuts(object(), reused, prompt_len)
            assert cuts == sorted(set(cuts)) and cuts[-1] == prompt_len and cuts[0] > reused


def test_every_piece_is_committed_to_the_slot(llama: LlamaArchitecture) -> None:
    cache = PromptCache()
    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    tokens = _engine(llama, cache, 4).generate(request).token_ids
    assert cache._slots[0].token_ids[: len(PROMPT)] == PROMPT
    assert len(cache._slots[0].token_ids) == len(PROMPT) + len(tokens)


def test_resume_after_cancel_matches_cold_run(llama: LlamaArchitecture) -> None:
    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    cold = _engine(llama, PromptCache(), 512).generate(request).token_ids
    for stop_after in (1, llama.n_layer, 2 * llama.n_layer + 1, 3 * llama.n_layer):
        engine = _engine(llama, PromptCache(), 4)
        list(engine.stream(request, _stop_after(stop_after)))
        assert engine.generate(request).token_ids == cold


def test_repeated_cancels_keep_making_progress(llama: LlamaArchitecture) -> None:
    cache = PromptCache()
    request = GenerationRequest(torch.tensor([PROMPT]), GREEDY)
    engine = _engine(llama, cache, 4)
    reused = []
    for _ in range(3):
        list(engine.stream(request, _stop_after(llama.n_layer + 1)))
        reused.append(sum(len(s.token_ids) for s in cache._slots))
    assert reused == sorted(reused) and reused[-1] > reused[0]
