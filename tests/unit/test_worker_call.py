"""WorkerCall: short jobs (embedding, reranking) run on the model's worker, which counts as busy, so
keep-alive expiry can't close a model mid-job (a reranker evicted while scoring crashed with
`BufferError: cannot close exported pointers exist`)."""

import asyncio
import threading
from collections.abc import Iterator

import pytest

from app.models.worker import ModelWorker
from app.models.worker_call import WorkerCall


@pytest.mark.asyncio
async def test_collect_returns_every_item_in_order() -> None:
    worker = ModelWorker()
    assert await WorkerCall.collect(worker, lambda: iter([0.5, 0.1, 0.9])) == [0.5, 0.1, 0.9]
    await asyncio.to_thread(worker.wait_until_idle)
    assert not worker.is_busy()


@pytest.mark.asyncio
async def test_the_worker_is_busy_while_the_job_runs() -> None:
    worker = ModelWorker()
    started, release = threading.Event(), threading.Event()

    def job() -> Iterator[int]:
        started.set()
        release.wait(5)
        yield 1

    task = asyncio.create_task(WorkerCall.collect(worker, job))
    await asyncio.to_thread(started.wait, 5)
    # What ModelManager._evict_expired checks before closing a model.
    assert worker.is_busy()
    release.set()
    assert await task == [1]
    await asyncio.to_thread(worker.wait_until_idle)
    assert not worker.is_busy()


@pytest.mark.asyncio
async def test_a_job_error_is_raised_to_the_caller() -> None:
    worker = ModelWorker()

    def job() -> Iterator[int]:
        yield 1
        raise ValueError("too long")

    with pytest.raises(ValueError, match="too long"):
        await WorkerCall.collect(worker, job)
    # The job's final "done" message comes after the error: let it finish before the loop closes.
    await asyncio.to_thread(worker.wait_until_idle)
    assert not worker.is_busy()


def test_the_worker_survives_a_caller_whose_loop_closed() -> None:
    """A result sent to a closed event loop used to raise in the worker thread and end it."""
    worker = ModelWorker()
    loop_closed = threading.Event()

    def job() -> Iterator[int]:
        loop_closed.wait(5)  # the caller's loop is gone before anything is sent
        yield 1

    loop = asyncio.new_event_loop()
    loop.run_until_complete(_submit(worker, job))
    loop.close()
    loop_closed.set()
    assert worker._idle_event.wait(5), "the worker thread died: the model would stay busy forever"
    assert worker._thread.is_alive()
    assert asyncio.run(WorkerCall.collect(worker, lambda: iter([3]))) == [3]


async def _submit(worker: ModelWorker, generator_fn) -> "asyncio.Queue[object]":
    return worker.stream(generator_fn)
