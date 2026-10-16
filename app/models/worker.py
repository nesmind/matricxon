import asyncio
import logging
import queue
import threading
from collections.abc import Callable, Iterator

import torch

from app.models.worker_job import WorkerJob
from app.runtime.batch_decode import BatchDecoder

logger = logging.getLogger(__name__)


class ModelWorker:
    """One dedicated background thread per loaded model, running every reply on it.

    Why: blocking PyTorch CPU compute never stalls FastAPI's event loop (a chat stream can run many
    seconds while the loop answers `/api/tags` health probes), and the model's caches and weights
    are touched from one thread, so forward passes can't race.

    Replies are generators (`ChatEngine.stream_steps`) advanced cooperatively. A reply asks for its
    newest token's logits by yielding a `DecodeStep`; each round the worker gives all active
    replies' pending steps to the `BatchDecoder`, which runs one forward pass when the architecture
    allows (each weight read once for all users: where concurrent throughput comes from), else one
    by one. A long prompt yields `CHECKPOINT` between prefill pieces so it can't stall others'
    decoding. At most `max_active` replies run at once; the rest queue in order. A plain generator
    yielding items (tests, `/api/generate`) runs on the same loop, one item per round.

    `app.server.thread_stream.stream_in_thread` is the same thread-to-asyncio bridge without the
    persistent thread, for one-off jobs (e.g. a `/api/pull` download).
    """

    def __init__(self, batch_decoder: BatchDecoder | None = None, max_active: int = 8) -> None:
        self._batch_decoder = batch_decoder
        self._max_active = max(1, max_active)
        # None is the shutdown sentinel (`shutdown`).
        self._jobs: queue.Queue[WorkerJob | None] = queue.Queue()
        self._stop_event = threading.Event()
        # One stop event per submitted reply (keyed by the caller's request id): cancelling one
        # never touches another user's reply on this model.
        self._job_stops: dict[str, threading.Event] = {}
        self._busy_count = 0
        self._busy_lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()
        # Called after each job ends; ModelManager restarts the keep-alive there, so it counts from
        # a reply's end, not its start (a reply longer than keep-alive unloaded the model at once).
        self.on_job_done: Callable[[], None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # state other threads read

    def set_max_active(self, max_active: int) -> None:
        """Concurrent replies, applied from the next scheduling round."""
        self._max_active = max(1, max_active)
        if self._batch_decoder is not None:
            self._batch_decoder.set_max_batch(max_active)

    @property
    def busy_count(self) -> int:
        """Replies queued or running on this worker now (GET /api/node/load)."""
        with self._busy_lock:
            return self._busy_count

    def is_busy(self) -> bool:
        """True while a stream job is queued or running on this worker.

        ModelManager checks it before evicting a handle (keep-alive expiry or LRU pressure): an
        evicted-but-generating handle used to become invisible to /api/ps and unreachable by
        request_stop while its thread kept computing. A slow generation routinely outlives the 300s
        default keep-alive, so the next unrelated get_or_load orphaned it mid-stream. See
        ModelManager._evict_expired/_evict_least_recently_used.
        """
        with self._busy_lock:
            return self._busy_count > 0

    def should_stop(self) -> bool:
        """The worker-wide stop flag (`request_stop`), for a generator's `stop_check`: lets a stop
        take effect *during* a forward pass, not just between the scheduler's checks (gap:
        cancelling mid-forward-pass on slow hardware could wait out a whole layer
        loop)."""
        return self._stop_event.is_set()

    def _job_stop_event(self, request_id: str) -> threading.Event:
        with self._busy_lock:
            return self._job_stops.setdefault(request_id, threading.Event())

    def stop_check_for(self, request_id: str | None) -> Callable[[], bool]:
        """The `stop_check` for one reply: true once *that* reply was cancelled (`cancel_job`) or
        the worker-wide `request_stop` was used (an explicit unload). Without an id:
        `should_stop`."""
        if request_id is None:
            return self.should_stop
        event = self._job_stop_event(request_id)
        return lambda: event.is_set() or self._stop_event.is_set()

    def cancel_job(self, request_id: str) -> bool:
        """Stops only the reply submitted under `request_id`: a running one halts at its next
        layer/token check, a queued one is skipped when its turn comes. Other replies and the loaded
        model are untouched. False if no such reply is running or queued here."""
        with self._busy_lock:
            event = self._job_stops.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def request_stop(self) -> None:
        """Stops every reply on this worker as soon as it next yields: the explicit "unload this
        model" path (`ModelManager.unload`), not routine eviction, which never interrupts a
        generation. A flag is enough: a reply the scheduler stops is closed and never resumed, so
        its further tokens' compute never starts."""
        self._stop_event.set()

    def wait_until_idle(self) -> None:
        """Blocks until no stream job is queued or running here. request_stop's flag isn't enough
        alone: it is asynchronous, noticed at the next per-layer/per-token check. Bug:
        `ModelManager.unload` called `request_stop` then `architecture.close` on another
        thread while this worker could be mid-`forward` reading a `QuantizedLinear`'s raw-mmap
        view; `mmap.close` crashed with `BufferError: cannot close exported pointers exist`."""
        self._idle_event.wait()

    # submitting a reply

    def stream(
        self, generator_fn: Callable[[], Iterator[object]], request_id: str | None = None
    ) -> "asyncio.Queue[object]":
        """Runs `generator_fn` (a generator) on this worker's thread. Each yielded item, then a
        final `None` sentinel (or any exception raised instead), reaches the calling coroutine's
        event loop via `call_soon_threadsafe`, readable with `await queue.get`.
        `DecodeStep`/`Checkpoint` items are the scheduler's protocol and never reach the queue."""
        loop = asyncio.get_running_loop()
        results: asyncio.Queue[object] = asyncio.Queue()
        stop_event = self._job_stop_event(request_id) if request_id is not None else None
        # Marked busy at submission, not when the thread picks the job up: is_busy must be true
        # once a caller committed to this job, including while it queues, or a manager eviction
        # check racing that window could treat this handle as idle and evict it.
        with self._busy_lock:
            self._busy_count += 1
            self._idle_event.clear()
        self._jobs.put(WorkerJob(generator_fn, loop, results, request_id, stop_event))
        return results

    # the scheduler (worker thread)

    def _run(self) -> None:
        # Gradients off for this thread for good: replies interleave here, so no grad-mode context
        # manager may span a suspension (ChatEngine.stream_steps).
        torch.set_grad_enabled(False)
        active: list[WorkerJob] = []
        closing = False
        while True:
            if not active:
                if closing:
                    break
                job = self._jobs.get()
                if job is None:
                    break
                # A stop aimed at replies that already finished must not kill this new one.
                self._stop_event.clear()
                active.append(job)
            while len(active) < self._max_active:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is None:
                    closing = True
                    break
                active.append(job)
            if self._stop_event.is_set():
                for job in active:
                    self._finish(job)
                self._stop_event.clear()
            for job in active:
                self._advance(job)
            self._decode_round([job for job in active if job.waiting is not None])
            active = [job for job in active if not job.done]
        # Drops the references to the model, so a retired one can be freed.
        self._batch_decoder = None
        self.on_job_done = None

    def shutdown(self) -> None:
        """Ends this worker's thread once its jobs are done (the model is being retired). The
        thread held the model (via its decoder and `on_job_done`), so without this every unloaded
        model stayed in RAM."""
        self.on_job_done = None
        self._jobs.put(None)

    def _stopped(self, job: WorkerJob) -> bool:
        return self._stop_event.is_set() or (job.stop_event is not None and job.stop_event.is_set())

    def _advance(self, job: WorkerJob) -> None:
        """Runs `job` until it needs logits, yields a checkpoint, or ends."""
        if job.advance(lambda: self._stopped(job)):
            self._finish(job)

    def _decode_round(self, waiting: list[WorkerJob]) -> None:
        """One batched decode step for every reply waiting on logits."""
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

    def _finish(self, job: WorkerJob) -> None:
        """Closes `job` and frees its place."""
        if not job.close():
            return
        with self._busy_lock:
            if job.request_id is not None:
                self._job_stops.pop(job.request_id, None)
            self._busy_count -= 1
            if self._busy_count == 0:
                self._idle_event.set()
        if self.on_job_done is not None:
            try:
                self.on_job_done()
            except Exception:  # noqa: BLE001 - must not kill the worker thread
                logger.exception("on_job_done callback failed")
