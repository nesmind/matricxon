import asyncio
import logging
import threading
from collections.abc import Callable, Iterator

from app.runtime.batch_decode import Checkpoint, DecodeStep

logger = logging.getLogger(__name__)


class WorkerJob:
    """One submitted reply on a `ModelWorker`: its generator, where its items go, and its place in
    the scheduler."""

    def __init__(
        self,
        generator_fn: Callable[[], Iterator[object]],
        loop: asyncio.AbstractEventLoop,
        results: "asyncio.Queue[object]",
        request_id: str | None,
        stop_event: threading.Event | None,
    ) -> None:
        self.generator_fn = generator_fn
        self.loop = loop
        self.results = results
        self.request_id = request_id
        self.stop_event = stop_event
        self.iterator: Iterator[object] | None = None
        self.waiting: DecodeStep | None = None
        self.reply: object = None
        self.done = False

    def emit(self, item: object) -> None:
        """Hands `item` to the caller's event loop. A closed loop (the caller is gone) is ignored:
        raising here would end the worker thread, and with it every later reply on the model."""
        try:
            self.loop.call_soon_threadsafe(self.results.put_nowait, item)
        except RuntimeError:
            pass

    def advance(self, stopped: Callable[[], bool]) -> bool:
        """Runs the generator until it needs logits, yields a checkpoint, or ends. True when it
        ended (stopped, finished or raised) and must be closed."""
        while self.waiting is None:
            if stopped():
                return True
            try:
                reply, self.reply = self.reply, None
                if self.iterator is None:
                    self.iterator = iter(self.generator_fn())
                    item = next(self.iterator)
                elif isinstance(reply, BaseException):
                    item = self.iterator.throw(reply)
                elif reply is None:
                    item = next(self.iterator)
                else:
                    item = self.iterator.send(reply)
            except StopIteration:
                return True
            except Exception as exc:  # noqa: BLE001 - handed to the async consumer, not swallowed
                self.emit(exc)
                return True
            if isinstance(item, DecodeStep):
                self.waiting = item
            elif isinstance(item, Checkpoint):
                return False
            else:
                self.emit(item)
        return False

    def close(self) -> bool:
        """Closes the generator (running its `finally` blocks: cache release) and tells the
        consumer. False if it was already closed."""
        if self.done:
            return False
        self.done = True
        self.waiting = None
        if self.iterator is not None and hasattr(self.iterator, "close"):
            try:
                self.iterator.close()
            except Exception:  # noqa: BLE001 - a failing cleanup must not kill the worker thread
                logger.exception("closing a reply's generator failed")
        self.emit(None)
        return True
