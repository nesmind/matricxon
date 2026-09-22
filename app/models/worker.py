import asyncio
import queue
import threading
from collections.abc import Callable, Iterator


class ModelWorker:
    """One dedicated background thread per loaded model.

    Two things this buys, per ROADMAP.md's M6 design: PyTorch's blocking CPU
    compute never stalls FastAPI's event loop (a chat stream can run for
    many seconds, and the loop also has to keep answering `/api/tags`
    health probes), and two forward passes against the same model's
    `KVCache` can never race each other - this worker's queue processes one
    job at a time, so concurrent requests against one model naturally
    serialize instead of corrupting shared state.

    See `app.server.thread_stream.stream_in_thread` for the same
    thread-to-asyncio bridge without the persistent-worker-thread part - a
    one-off job with no shared state to protect (e.g. a `/api/pull`
    download) uses that directly instead of a whole `ModelWorker`.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._stop_event = threading.Event()
        self._busy_count = 0
        self._busy_lock = threading.Lock()
        self._idle_event = threading.Event()
        self._idle_event.set()

    def is_busy(self) -> bool:
        """True while at least one stream() job is queued or actively running on this worker.

        ModelManager checks this before evicting a handle (keep-alive expiry or LRU pressure) - an
        evicted-but-still-generating handle used to become invisible to /api/ps and unreachable by
        request_stop() (ModelManager.unload() only knew how to signal a handle still in its own
        dict), while its worker thread kept right on computing every remaining token regardless.
        Confirmed live: a single slow generation (num_predict defaults to 1024, and matricxon's own
        per-token cost dwarfs Ollama's) routinely outlives the 300s default keep-alive, so the very
        next unrelated request's own get_or_load() call would silently orphan it mid-stream. See
        ModelManager._evict_expired/_evict_least_recently_used's own comments.
        """
        with self._busy_lock:
            return self._busy_count > 0

    def should_stop(self) -> bool:
        """Exposes `_stop_event` read-only, for a caller building the `generator_fn` passed to
        `stream()` (see `app.runtime.chat_engine.ChatEngine.stream`'s own `stop_check` parameter)
        to check *during* a single generation step, not just between the items `stream()`'s own
        loop below already checks between. Real gap this closes (2026-09-21, confirmed live): a
        stop previously only took effect between tokens, so cancelling mid-forward-pass on this
        project's slow hardware could still mean waiting out an entire in-progress layer loop -
        many seconds to minutes on a large prompt - before it had any visible effect."""
        return self._stop_event.is_set()

    def request_stop(self) -> None:
        """Stops whatever job is currently running (or about to run) on this worker as soon as it
        next yields, instead of letting it grind on to max_new_tokens/EOS unseen - called from
        ModelManager.unload() when a caller explicitly wants generation aborted (e.g. pAIring's own
        "stop_model" unload call after a mid-stream message delete), not from routine LRU/keep-alive
        eviction, which has no reason to interrupt an in-flight generation.

        Setting a flag is enough because `stream()`'s consumer loop below only pulls the *next*
        item from `generator_fn()` when it's about to keep going - once it stops iterating, a
        Python generator paused at its last `yield` never resumes, so the underlying (in this
        case, PyTorch) compute for any further tokens never even starts. No forced thread
        interruption needed.
        """
        self._stop_event.set()

    def wait_until_idle(self) -> None:
        """Blocks until no stream() job is queued or actively running on this worker - see
        request_stop's own docstring for why setting the stop flag isn't enough by itself: it's
        asynchronous, only noticed at the next per-layer/per-token check, not instant. A caller
        that needs the worker's in-flight forward pass to have genuinely finished - not just been
        asked to - before touching anything the forward pass might still be using has to wait for
        real, not just request and move on. Real bug this closes (2026-09-22, confirmed live):
        `ModelManager.unload` called `request_stop()` then `architecture.close()` right after, on
        a different thread, with nothing in between - `close()` released every real
        `QuantizedLinear`'s live raw-mmap view and closed the mmap itself while this worker's own
        thread could still be mid-`forward()`, actively reading that same raw memoryview inside a
        fused GEMV kernel call. `mmap.close()` crashed with a real `BufferError: cannot close
        exported pointers exist` - the exact race `QuantizedLinear.release`'s own docstring
        already warned about, just never actually closed here until now.
        """
        self._idle_event.wait()

    def stream(self, generator_fn: Callable[[], Iterator[object]]) -> "asyncio.Queue[object]":
        """Runs `generator_fn()` (a synchronous generator) on this worker's
        thread. Each yielded item, then a final `None` sentinel (or any
        exception raised instead), is handed to the calling coroutine's
        event loop via `call_soon_threadsafe` and can be read off the
        returned queue with `await queue.get()`.
        """
        loop = asyncio.get_running_loop()
        results: asyncio.Queue[object] = asyncio.Queue()

        # Marked busy here, at submission time (not once the worker thread actually picks the job
        # up) - is_busy() has to read "true" the instant a caller has committed to running this job,
        # including the window while it's still sitting in the queue behind another one, or a
        # manager-side eviction check racing that window could still wrongly treat this handle as
        # idle and evict it.
        with self._busy_lock:
            self._busy_count += 1
            self._idle_event.clear()

        def produce() -> None:
            # Cleared here - right as this job actually starts running, not back when stream() was
            # merely called - so a stop requested against an *earlier* job on this same worker (this
            # queue can hold more than one: concurrent requests against the same model tag serialize
            # onto it, see this class's own docstring) can never leak forward and instantly kill a
            # later job that hasn't even started yet. The flip side: a stop requested while THIS job
            # is still queued (not yet dequeued) gets cleared away right here and has no effect on
            # it - acceptable, since nothing is actually running/burning CPU for it to interrupt
            # until it's the one being consumed below; request_stop's real job is cutting off an
            # *already-executing* generation, which this still does correctly.
            self._stop_event.clear()
            try:
                for item in generator_fn():
                    loop.call_soon_threadsafe(results.put_nowait, item)
                    if self._stop_event.is_set():
                        break
            except Exception as exc:  # noqa: BLE001 - handed to the async consumer, not swallowed
                loop.call_soon_threadsafe(results.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(results.put_nowait, None)
                with self._busy_lock:
                    self._busy_count -= 1
                    if self._busy_count == 0:
                        self._idle_event.set()

        self._jobs.put(produce)
        return results

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            job()
