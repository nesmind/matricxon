import asyncio
import threading

import pytest

from app.models.worker import ModelWorker


async def _drain(results: "asyncio.Queue[object]") -> list[object]:
    items: list[object] = []
    while True:
        item = await results.get()
        if item is None:
            return items
        items.append(item)


class TestModelWorkerStream:
    @pytest.mark.asyncio
    async def test_runs_the_generator_to_completion_when_never_stopped(self) -> None:
        worker = ModelWorker()

        def gen():
            yield from range(5)

        items = await _drain(worker.stream(gen))
        assert items == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_propagates_an_exception_raised_by_the_generator(self) -> None:
        worker = ModelWorker()

        def gen():
            yield 1
            raise ValueError("boom")

        results = worker.stream(gen)
        first = await results.get()
        assert first == 1
        exc = await results.get()
        assert isinstance(exc, ValueError)


class TestModelWorkerRequestStop:
    @pytest.mark.asyncio
    async def test_request_stop_eventually_halts_the_generator(self) -> None:
        """The point of request_stop: once the flag is seen, the generator is never resumed. Proven
        with a counter the generator increments on each resume (a stand-in for per-token compute).
        How many items are in flight when the produce loop's is_set check runs is a race between
        the worker thread and the thread calling request_stop: checked-after-each-item cooperative
        cancellation only *bounds* it (varies run to run). What's proven is the guarantee that
        matters: the generator never gets near completing (1000 items) after a stop."""
        worker = ModelWorker()
        resumed_count = 0
        proceed = threading.Event()

        def gen():
            nonlocal resumed_count
            for i in range(1000):
                resumed_count += 1
                yield i
                if i == 0:
                    proceed.wait()

        results = worker.stream(gen)
        first = await results.get()
        assert first == 0

        worker.request_stop()
        proceed.set()

        remaining = await _drain(results)
        # At most one more item could be "in flight": whether 0 (is_set right after item 0 saw it)
        # or 1 (item 1 already computed) is the race above.
        assert remaining in ([], [1])
        assert resumed_count in (1, 2)
        assert resumed_count == 2

    @pytest.mark.asyncio
    async def test_a_stop_against_a_finished_job_does_not_leak_into_the_next_one(self) -> None:
        """Guard: stream cleared the stop flag when *submitted*, not when the job started, so a
        stop for job A after it finished (but before job B queued behind it started) instantly
        killed B though nobody asked."""
        worker = ModelWorker()

        def short_gen():
            yield "a"

        await _drain(worker.stream(short_gen))
        worker.request_stop()

        def long_gen():
            yield from ["b1", "b2", "b3"]

        items = await _drain(worker.stream(long_gen))
        assert items == ["b1", "b2", "b3"]


class TestModelWorkerWaitUntilIdle:
    def test_returns_immediately_when_nothing_has_ever_run(self) -> None:
        worker = ModelWorker()
        worker.wait_until_idle()

    @pytest.mark.asyncio
    async def test_blocks_until_the_in_flight_generator_actually_stops(self) -> None:
        """Guards a bug: ModelManager.unload called architecture.close right after
        request_stop with nothing ensuring the worker had finished its in-flight forward pass,
        crashing with `BufferError` when quantized-native compute was reading a live raw-mmap view
        (ModelManager.unload). wait_until_idle must still report busy while the worker is stuck
        mid-generator, not just between items the consumer drained."""
        worker = ModelWorker()
        reached_second_item = threading.Event()
        became_idle = threading.Event()

        def gen():
            yield 0
            reached_second_item.wait()
            yield 1

        results = worker.stream(gen)
        first = await results.get()
        assert first == 0

        def wait_in_background() -> None:
            worker.wait_until_idle()
            became_idle.set()

        waiter = threading.Thread(target=wait_in_background, daemon=True)
        waiter.start()

        # The worker thread is stuck inside gen now (past yield 0, blocked on
        # reached_second_item): wait_until_idle must not have returned yet.
        assert not became_idle.wait(timeout=0.2)

        reached_second_item.set()
        await _drain(results)
        waiter.join(timeout=1.0)
        assert became_idle.is_set()
