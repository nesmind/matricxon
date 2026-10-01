import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Iterator

import torch

from app.runtime.batch_decode import BatchDecoder, Checkpoint, DecodeStep

logger = logging.getLogger(__name__)


class _Job:
    """One submitted reply: its generator, where its items go, and its place in the scheduler."""

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
        self.waiting: DecodeStep | None = None  # wants logits - the next batched decode gives them
        self.reply: object = None  # what to send the generator when it next runs
        self.done = False

    def emit(self, item: object) -> None:
        self.loop.call_soon_threadsafe(self.results.put_nowait, item)


class ModelWorker:
    """One dedicated background thread per loaded model, running every reply on that model.

    Two things this buys, per ROADMAP.md's M6 design: PyTorch's blocking CPU compute never stalls
    FastAPI's event loop (a chat stream can run for many seconds, and the loop also has to keep
    answering `/api/tags` health probes), and the model's caches and weights are only ever touched
    from this one thread, so forward passes can never race each other.

    Replies are generators (`ChatEngine.stream_steps`) that this thread advances cooperatively.
    A reply asks for the logits of its newest token by yielding a `DecodeStep`; each round the
    worker collects the pending steps of all active replies and gives them to the `BatchDecoder`,
    which runs them as one forward pass when the architecture allows (each weight is then read once
    for every user, which is where concurrent throughput comes from) and one by one otherwise. A
    long prompt yields `CHECKPOINT` between prefill pieces so it cannot stall everyone else's
    decoding. At most `max_active` replies run at once; the rest wait in the queue, in order. A
    plain generator that yields only items - what the tests and `/api/generate` used before - runs
    on the same loop, one item per round.

    See `app.server.thread_stream.stream_in_thread` for the same thread-to-asyncio bridge without
    the persistent-worker-thread part - a one-off job with no shared state to protect (e.g. a
    `/api/pull` download) uses that directly instead of a whole `ModelWorker`.
    """

    def __init__(self, batch_decoder: BatchDecoder | None = None, max_active: int = 8) -> None:
        self._batch_decoder = batch_decoder
        self._max_active = max(1, max_active)
        self._jobs: queue.Queue[_Job] = queue.Queue()
        self._stop_event = threading.Event()
        # One stop event per submitted reply (keyed by the caller's request id): cancelling one
        # reply never touches another user's reply running or queued on this same model.
        self._job_stops: dict[str, threading.Event] = {}
        self._busy_count = 0
        self._busy_lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ---- state other threads read ------------------------------------------------------------

    def is_busy(self) -> bool:
        """True while at least one stream() job is queued or actively running on this worker.

        ModelManager checks this before evicting a handle (keep-alive expiry or LRU pressure) - an
        evicted-but-still-generating handle used to become invisible to /api/ps and unreachable by
        request_stop() (ModelManager.unload() only knew how to signal a handle still in its own
        dict), while its worker thread kept right on computing every remaining token regardless.
        Confirmed live: a single slow generation routinely outlives the 300s default keep-alive, so
        the very next unrelated request's own get_or_load() call would silently orphan it
        mid-stream. See ModelManager._evict_expired/_evict_least_recently_used's own comments.
        """
        with self._busy_lock:
            return self._busy_count > 0

    def should_stop(self) -> bool:
        """The worker-wide stop flag (`request_stop`), for a caller building the `stop_check` of
        a generator: lets a stop take effect *during* a single forward pass, not just between the
        items the scheduler checks between (real gap, 2026-09-21: cancelling mid-forward-pass on
        slow hardware could mean waiting out an entire layer loop)."""
        return self._stop_event.is_set()

    def _job_stop_event(self, request_id: str) -> threading.Event:
        with self._busy_lock:
            return self._job_stops.setdefault(request_id, threading.Event())

    def stop_check_for(self, request_id: str | None) -> Callable[[], bool]:
        """The `stop_check` for one reply: true once *that* reply was cancelled (`cancel_job`) or
        the worker-wide `request_stop` was used (an explicit unload). Without an id: `should_stop`.
        """
        if request_id is None:
            return self.should_stop
        event = self._job_stop_event(request_id)
        return lambda: event.is_set() or self._stop_event.is_set()

    def cancel_job(self, request_id: str) -> bool:
        """Stops only the reply submitted under `request_id`: a running one halts at its next
        layer/token check, a queued one is skipped when its turn comes. Other replies on this worker
        are untouched and the model stays loaded. False if no such reply is running or queued here.
        """
        with self._busy_lock:
            event = self._job_stops.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def request_stop(self) -> None:
        """Stops every reply running on this worker as soon as it next yields - the explicit
        "unload this model" path (`ModelManager.unload`), not routine eviction, which never
        interrupts a generation. Setting a flag is enough: a reply the scheduler stops is closed
        and never resumed, so its further tokens' compute never even starts."""
        self._stop_event.set()

    def wait_until_idle(self) -> None:
        """Blocks until no stream() job is queued or actively running on this worker - see
        request_stop's own docstring for why setting the stop flag isn't enough by itself: it's
        asynchronous, only noticed at the next per-layer/per-token check, not instant. Real bug this
        closes (2026-09-22, confirmed live): `ModelManager.unload` called `request_stop()` then
        `architecture.close()` right after, on a different thread, while this worker could still be
        mid-`forward()` reading a `QuantizedLinear`'s live raw-mmap view - `mmap.close()` crashed
        with `BufferError: cannot close exported pointers exist`."""
        self._idle_event.wait()

    # ---- submitting a reply ------------------------------------------------------------------

    def stream(
        self, generator_fn: Callable[[], Iterator[object]], request_id: str | None = None
    ) -> "asyncio.Queue[object]":
        """Runs `generator_fn()` (a generator) on this worker's thread. Each yielded item, then a
        final `None` sentinel (or any exception raised instead), is handed to the calling
        coroutine's event loop via `call_soon_threadsafe` and can be read off the returned queue
        with `await queue.get()`. `DecodeStep`/`Checkpoint` items are the scheduler's own protocol
        and never reach the queue.
        """
        loop = asyncio.get_running_loop()
        results: asyncio.Queue[object] = asyncio.Queue()
        stop_event = self._job_stop_event(request_id) if request_id is not None else None
        # Marked busy here, at submission time (not once the thread actually picks the job up) -
        # is_busy() has to read "true" the instant a caller has committed to running this job,
        # including the window while it's still sitting in the queue, or a manager-side eviction
        # check racing that window could still wrongly treat this handle as idle and evict it.
        with self._busy_lock:
            self._busy_count += 1
            self._idle_event.clear()
        self._jobs.put(_Job(generator_fn, loop, results, request_id, stop_event))
        return results

    # ---- the scheduler (worker thread) ---------------------------------------------------------

    def _run(self) -> None:
        # Gradients off for this thread for good: replies interleave here, so no grad-mode context
        # manager may span a suspension (see ChatEngine.stream_steps).
        torch.set_grad_enabled(False)
        active: list[_Job] = []
        while True:
            if not active:
                job = self._jobs.get()
                # A stop aimed at replies that already finished must not kill this new one.
                self._stop_event.clear()
                active.append(job)
            while len(active) < self._max_active:
                try:
                    active.append(self._jobs.get_nowait())
                except queue.Empty:
                    break
            if self._stop_event.is_set():
                for job in active:
                    self._finish(job)
                self._stop_event.clear()
            for job in active:
                self._advance(job)
            self._decode_round([job for job in active if job.waiting is not None])
            active = [job for job in active if not job.done]

    def _stopped(self, job: _Job) -> bool:
        return self._stop_event.is_set() or (job.stop_event is not None and job.stop_event.is_set())

    def _advance(self, job: _Job) -> None:
        """Runs `job` until it needs logits, yields a checkpoint, or ends."""
        while not job.done and job.waiting is None:
            if self._stopped(job):
                self._finish(job)  # cancelled (also while still queued): nothing more computed
                return
            try:
                reply, job.reply = job.reply, None
                if job.iterator is None:
                    job.iterator = iter(job.generator_fn())
                    item = next(job.iterator)
                elif isinstance(reply, BaseException):
                    item = job.iterator.throw(reply)
                elif reply is None:
                    item = next(job.iterator)
                else:
                    item = job.iterator.send(reply)
            except StopIteration:
                self._finish(job)
                return
            except Exception as exc:  # noqa: BLE001 - handed to the async consumer, not swallowed
                job.emit(exc)
                self._finish(job)
                return
            if isinstance(item, DecodeStep):
                job.waiting = item
            elif isinstance(item, Checkpoint):
                return
            else:
                job.emit(item)

    def _decode_round(self, waiting: list[_Job]) -> None:
        """One batched decode step for every reply that is waiting on logits."""
        waiting = [job for job in waiting if not job.done]
        if not waiting:
            return
        steps = [job.waiting for job in waiting]
        try:
            if self._batch_decoder is None:
                raise RuntimeError("this worker has no batch decoder to run decode steps with")
            results: list[object] = list(self._batch_decoder.decode(steps))
        except Exception as exc:  # noqa: BLE001 - delivered to each waiting reply
            results = [exc] * len(waiting)
        for job, result in zip(waiting, results, strict=True):
            job.waiting = None
            job.reply = result

    def _finish(self, job: _Job) -> None:
        """Closes `job`'s generator (running its `finally` blocks - cache release), tells the
        consumer, and frees its place."""
        if job.done:
            return
        job.done = True
        job.waiting = None
        if job.iterator is not None and hasattr(job.iterator, "close"):
            try:
                job.iterator.close()
            except Exception:  # noqa: BLE001 - a failing cleanup must not kill the worker thread
                logger.exception("closing a reply's generator failed")
        job.emit(None)
        with self._busy_lock:
            if job.request_id is not None:
                self._job_stops.pop(job.request_id, None)
            self._busy_count -= 1
            if self._busy_count == 0:
                self._idle_event.set()
