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
        """The whole point of request_stop: once the flag is seen, the generator is never resumed
        again - proven here with a counter the generator itself increments on every resume (a real
        stand-in for real per-token compute). Exactly how many items are already "in flight"
        (already computed, already scheduled for delivery to the consumer) at the moment the
        produce() loop's own is_set() check actually runs is a genuine race between the worker
        thread and whichever thread calls request_stop() - checked-after-each-item cooperative
        cancellation only ever *bounds* that count, it can never pin it to one specific number
        (confirmed live: this really does vary run to run). What's actually being proven here is
        the guarantee that matters: the generator never gets anywhere close to running to
        completion (1000 items) once a stop has been requested."""
        worker = ModelWorker()
        resumed_count = 0
        proceed = threading.Event()

        def gen():
            nonlocal resumed_count
            for i in range(1000):
                resumed_count += 1
                yield i
                if i == 0:
                    proceed.wait()  # holds the worker thread here until the test says go

        results = worker.stream(gen)
        first = await results.get()
        assert first == 0

        worker.request_stop()
        proceed.set()  # let the worker continue past item 0, however far it gets before noticing

        remaining = await _drain(results)
        # At most one more item could possibly have been "in flight" - whether that's 0 (the
        # is_set() check right after item 0 already saw it) or 1 (item 1 was already computed
        # first) is the race described above, not something this test can pin down further.
        assert remaining in ([], [1])
        assert resumed_count in (1, 2)
        assert resumed_count == 2

    @pytest.mark.asyncio
    async def test_a_stop_against_a_finished_job_does_not_leak_into_the_next_one(self) -> None:
        """Regression guard: stream() used to clear the stop flag when *submitted*, not when the
        job actually started running - a stop requested for job A after it finished (but before job
        B, queued right behind it on the same worker, started) would otherwise instantly kill B
        too, even though nobody asked to stop B."""
        worker = ModelWorker()

        def short_gen():
            yield "a"

        await _drain(worker.stream(short_gen))  # job A runs to completion and finishes
        worker.request_stop()  # a (now-meaningless) stop for the already-finished job A

        def long_gen():
            yield from ["b1", "b2", "b3"]

        items = await _drain(worker.stream(long_gen))
        assert items == ["b1", "b2", "b3"]


class TestModelWorkerWaitUntilIdle:
    def test_returns_immediately_when_nothing_has_ever_run(self) -> None:
        worker = ModelWorker()
        worker.wait_until_idle()  # must not block

    @pytest.mark.asyncio
    async def test_blocks_until_the_in_flight_generator_actually_stops(self) -> None:
        """Real bug this guards against (2026-09-22): ModelManager.unload used to call
        architecture.close() right after request_stop(), with nothing making sure the worker's own
        thread had actually finished its in-flight forward pass first - a real race that crashed
        with a `BufferError` when real quantized-native compute was reading a live raw-mmap view
        at that exact moment (see ModelManager.unload's own docstring). wait_until_idle() must
        still report busy while the worker thread is genuinely stuck mid-generator, not just
        between items the consumer has already drained."""
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

        # The worker thread is genuinely stuck inside gen() right now (past yield 0, blocked on
        # reached_second_item) - wait_until_idle() must not have returned yet.
        assert not became_idle.wait(timeout=0.2)

        reached_second_item.set()
        await _drain(results)
        waiter.join(timeout=1.0)
        assert became_idle.is_set()
