from collections.abc import Callable, Iterator
from typing import TypeVar

from app.models.worker import ModelWorker

T = TypeVar("T")


class WorkerCall:
    """Runs a short job (embedding, reranking) on a model's worker thread and returns its items.

    On the worker the model counts as busy, so keep-alive expiry never closes it mid-job (it did:
    a reranker evicted while scoring crashed with `BufferError: cannot close exported pointers
    exist`), an unload waits for the job, and the weights are touched from one thread only.
    """

    @staticmethod
    async def collect(worker: ModelWorker, generator_fn: Callable[[], Iterator[T]]) -> list[T]:
        """Every item `generator_fn` yields, in order; re-raises the job's exception."""
        results = worker.stream(generator_fn)
        items: list[T] = []
        while (item := await results.get()) is not None:
            if isinstance(item, Exception):
                raise item
            items.append(item)
        return items
