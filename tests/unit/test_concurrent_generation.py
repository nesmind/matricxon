"""Replies running at the same time on one loaded model (a real tiny model, the real scheduler and
batch decoder) must come out exactly as if each had run alone - and their decode steps really are
batched. Both a plain KV-cache model and the hybrid Qwen 3.5 (recurrent state per sequence)."""

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.architectures.qwen35 import Qwen35Architecture
from app.gguf.loader import GGUFModelLoader
from app.models.worker import ModelWorker
from app.runtime.batch_decode import BatchDecoder
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf
from tests.tiny_gguf_qwen35 import build_tiny_qwen35_gguf

SAMPLING = SamplingConfig(temperature=0.0, num_predict=6, num_ctx=96)
PROMPTS = [[1, 72, 105, 33, 90, 44], [1, 200, 201, 202], [1, 60, 61, 62, 63, 64, 65, 66]]


class RecordingDecoder(BatchDecoder):
    def __init__(self, architecture) -> None:
        super().__init__(architecture)
        self.batch_sizes: list[int] = []

    def decode(self, steps):
        self.batch_sizes.append(len(steps))
        return super().decode(steps)


def _alone(model, ids: list[int]) -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=PromptCache())
    return engine.generate(GenerationRequest(torch.tensor([ids]), SAMPLING)).token_ids


async def _concurrently(
    model, prompts: list[list[int]], chunk: int = 512
) -> tuple[list, list[int]]:
    decoder = RecordingDecoder(model)
    worker = ModelWorker(decoder, max_active=8)
    pool = PromptCache()
    streams = []
    for ids in prompts:
        engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool, prefill_chunk=chunk)
        request = GenerationRequest(torch.tensor([ids]), SAMPLING)
        streams.append(worker.stream(lambda e=engine, r=request: e.stream_steps(r)))
    outputs = []
    for results in streams:
        items = []
        while (item := await results.get()) is not None:
            assert not isinstance(item, Exception), item
            items.append(item)
        outputs.append(items)
    return outputs, decoder.batch_sizes


def _load(builder: Callable[[Path], Path], cls, tmp_path: Path) -> Iterator:
    path = builder(tmp_path / "m.gguf")
    return cls.from_gguf(GGUFModelLoader(path, dtype=torch.float32)).eval()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("builder", "cls", "chunk"),
    [
        (build_tiny_llama_gguf, LlamaArchitecture, 512),
        (build_tiny_qwen35_gguf, Qwen35Architecture, 512),
        (build_tiny_qwen35_gguf, Qwen35Architecture, 3),  # chunked prefill interleaves too
    ],
    ids=["llama", "qwen35", "qwen35-chunked"],
)
async def test_concurrent_replies_equal_running_alone_and_decode_in_batches(
    builder, cls, chunk, tmp_path: Path
) -> None:
    model = _load(builder, cls, tmp_path)
    alone = [_alone(model, ids) for ids in PROMPTS]

    together, batch_sizes = await _concurrently(model, PROMPTS, chunk)

    assert together == alone
    assert max(batch_sizes) >= 2  # decode steps really were batched, not run one by one
    model.close()


@pytest.mark.asyncio
async def test_cancelling_one_concurrent_reply_leaves_the_others_exact(tmp_path: Path) -> None:
    model = _load(build_tiny_llama_gguf, LlamaArchitecture, tmp_path)
    alone = [_alone(model, ids) for ids in PROMPTS]
    worker = ModelWorker(BatchDecoder(model), max_active=8)
    pool = PromptCache()
    streams = []
    for i, ids in enumerate(PROMPTS):
        engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool)
        request = GenerationRequest(torch.tensor([ids]), SAMPLING)
        stop = worker.stop_check_for(f"r{i}")
        streams.append(
            worker.stream(lambda e=engine, r=request, s=stop: e.stream_steps(r, s), f"r{i}")
        )
    worker.cancel_job("r1")

    outputs = []
    for results in streams:
        items = []
        while (item := await results.get()) is not None:
            items.append(item)
        outputs.append(items)

    assert outputs[0] == alone[0] and outputs[2] == alone[2]
    assert len(outputs[1]) < len(alone[1]) or outputs[1] == []
    worker.wait_until_idle()
    model.close()
