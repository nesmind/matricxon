"""ModelWorker's scheduler with a fake decoder: the decode steps of replies running at the same
time arrive together, a cancel stays with its own reply, and the active cap holds."""

import asyncio
import threading

import pytest
import torch

from app.models.worker import ModelWorker
from app.runtime.batch_decode import CHECKPOINT, DecodeStep


class FakeDecoder:
    """Records each batch it is asked for; the 'logits' just echo the step's token."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []

    def decode(self, steps: list[DecodeStep]) -> list[torch.Tensor]:
        self.batches.append([s.token_id for s in steps])
        return [torch.tensor([float(s.token_id)]) for s in steps]


def reply(first: int, n: int, closed: list[int] | None = None):
    """A reply that asks for logits n times, yielding what it was sent."""
    try:
        for i in range(n):
            logits = yield DecodeStep(first + i, cache=None, position=i)
            yield int(logits[0])
    finally:
        if closed is not None:
            closed.append(first)


async def drain(results: "asyncio.Queue[object]") -> list[object]:
    items: list[object] = []
    while (item := await results.get()) is not None:
        items.append(item)
    return items


@pytest.mark.asyncio
async def test_decode_steps_of_replies_running_together_are_batched() -> None:
    decoder = FakeDecoder()
    worker = ModelWorker(decoder, max_active=4)
    gate = threading.Event()

    def held(first, n):
        gate.wait(5)  # both replies are queued before the scheduler starts either
        yield from reply(first, n)

    a = worker.stream(lambda: held(10, 3))
    b = worker.stream(lambda: held(100, 3))
    await asyncio.sleep(0.05)
    gate.set()

    assert await drain(a) == [10, 11, 12]
    assert await drain(b) == [100, 101, 102]
    assert decoder.batches == [[10, 100], [11, 101], [12, 102]]


@pytest.mark.asyncio
async def test_a_reply_that_finishes_early_leaves_the_others_unbatched_afterwards() -> None:
    decoder = FakeDecoder()
    worker = ModelWorker(decoder, max_active=4)
    gate = threading.Event()

    def held(first, n):
        gate.wait(5)
        yield from reply(first, n)

    short = worker.stream(lambda: held(1, 1))
    long = worker.stream(lambda: held(50, 3))
    await asyncio.sleep(0.05)
    gate.set()

    assert await drain(short) == [1]
    assert await drain(long) == [50, 51, 52]
    assert decoder.batches == [[1, 50], [51], [52]]


@pytest.mark.asyncio
async def test_cancelling_one_reply_closes_it_and_leaves_the_other_running() -> None:
    decoder = FakeDecoder()
    worker = ModelWorker(decoder, max_active=4)
    closed: list[int] = []
    gate = threading.Event()

    def held(first, n):
        gate.wait(5)
        yield from reply(first, n, closed)

    a = worker.stream(lambda: held(10, 50), request_id="a")
    b = worker.stream(lambda: held(100, 3), request_id="b")
    await asyncio.sleep(0.05)
    assert worker.cancel_job("a") is True
    gate.set()

    assert await drain(b) == [100, 101, 102]
    assert len(await drain(a)) < 50
    assert 10 in closed  # its generator was closed, so its cleanup (cache release) ran


@pytest.mark.asyncio
async def test_replies_beyond_the_active_cap_wait_their_turn() -> None:
    decoder = FakeDecoder()
    worker = ModelWorker(decoder, max_active=2)
    gate = threading.Event()

    def held(first, n):
        gate.wait(5)
        yield from reply(first, n)

    streams = [worker.stream(lambda f=f: held(f, 2)) for f in (10, 20, 30)]
    await asyncio.sleep(0.05)
    gate.set()

    for first, results in zip((10, 20, 30), streams, strict=True):
        assert await drain(results) == [first, first + 1]
    assert max(len(batch) for batch in decoder.batches) == 2  # never three at once
    assert [30, 31] == [t for batch in decoder.batches for t in batch if t >= 30]


@pytest.mark.asyncio
async def test_a_failing_decode_fails_only_the_reply_it_belongs_to() -> None:
    class PickyDecoder(FakeDecoder):
        def decode(self, steps):
            results = super().decode(steps)
            return [
                RuntimeError("boom") if s.token_id == 100 else r
                for s, r in zip(steps, results, strict=True)
            ]

    worker = ModelWorker(PickyDecoder(), max_active=4)
    gate = threading.Event()

    def held(first, n):
        gate.wait(5)
        yield from reply(first, n)

    good = worker.stream(lambda: held(10, 2))
    bad = worker.stream(lambda: held(100, 2))
    await asyncio.sleep(0.05)
    gate.set()

    assert await drain(good) == [10, 11]
    items = await drain(bad)
    assert len(items) == 1 and isinstance(items[0], RuntimeError)


@pytest.mark.asyncio
async def test_a_checkpoint_lets_another_reply_run_in_between() -> None:
    worker = ModelWorker(FakeDecoder(), max_active=4)
    order: list[str] = []
    gate = threading.Event()

    def prefill():
        gate.wait(5)
        for piece in range(3):
            order.append(f"a{piece}")
            yield CHECKPOINT
        yield "a-done"

    def quick():
        gate.wait(5)
        order.append("b")
        yield "b-done"

    a = worker.stream(prefill)
    b = worker.stream(quick)
    await asyncio.sleep(0.05)
    gate.set()
    await drain(a), await drain(b)

    assert order == ["a0", "b", "a1", "a2"]  # b ran after a's first piece, not after all of a
