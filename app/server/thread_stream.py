import asyncio
import threading
from collections.abc import Callable, Iterator


def stream_in_thread(generator_fn: Callable[[], Iterator[object]]) -> "asyncio.Queue[object]":
    """Runs a synchronous generator on a fresh background thread, bridging its output into the
    calling coroutine's event loop: each yielded item, then a final `None` sentinel (or any
    exception raised), is handed over via `call_soon_threadsafe` and read with `await queue.get`.
    The general "blocking producer, async consumer" bridge: `ModelWorker` builds on it (one
    persistent thread per loaded model, so forward passes against its `KVCache` serialize); a
    one-off job with no shared state (e.g. a `/api/pull` download) can use it directly."""
    loop = asyncio.get_running_loop()
    results: asyncio.Queue[object] = asyncio.Queue()

    def produce() -> None:
        try:
            for item in generator_fn():
                loop.call_soon_threadsafe(results.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 - handed to the async consumer, not swallowed
            loop.call_soon_threadsafe(results.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(results.put_nowait, None)

    threading.Thread(target=produce, daemon=True).start()
    return results
