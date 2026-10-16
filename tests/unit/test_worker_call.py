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
    assert not worker.is_busy()


@pytest.mark.asyncio
async def test_a_job_error_is_raised_to_the_caller() -> None:
    worker = ModelWorker()

    def job() -> Iterator[int]:
        yield 1
        raise ValueError("too long")

    with pytest.raises(ValueError, match="too long"):
        await WorkerCall.collect(worker, job)
    assert not worker.is_busy()
