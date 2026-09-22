import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class DownloadCoordinator:
    """Serializes concurrent `/api/pull` requests for the same destination file - real bug found
    via a pAIring report (2026-09-21): PullJob._run had no coordination between two in-flight
    pulls of the same tag (e.g. a retry while an earlier, now-orphaned attempt was still
    running), so both HFDownloader.download calls streamed to the same `<dest>.partial` path.
    Whichever finished first renamed it away to `dest`; the other then crashed the whole ASGI
    response with a raw `FileNotFoundError` on its own `partial_path.rename(dest)` - which is
    exactly what a client watching the SSE stream sees as a mid-stream connection drop, not a
    clean in-band `{"error": ...}` line the way every other pull failure here is handled.

    One `threading.Lock` per distinct `dest` (not one global lock across every pull - two
    different tags must still download fully in parallel, only the same one needs to wait its
    turn), created lazily and dropped again once nothing's waiting on it, so this never grows
    unbounded across a long-running process pulling many different models over its lifetime.
    """

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[Path, threading.Lock] = {}
        self._waiters: dict[Path, int] = defaultdict(int)

    @contextmanager
    def guard(self, dest: Path) -> Iterator[None]:
        """Blocks until no other caller is inside `guard(dest)` for this exact `dest`, then
        holds it until the `with` block exits (success or exception) - the caller decides what
        "already handled by someone else" means once it has the lock (see PullJob._run, which
        checks `dest.exists()` first)."""
        lock = self._acquire_lock_slot(dest)
        lock.acquire()
        try:
            yield
        finally:
            lock.release()
            self._release_lock_slot(dest)

    def _acquire_lock_slot(self, dest: Path) -> threading.Lock:
        with self._registry_lock:
            lock = self._locks.setdefault(dest, threading.Lock())
            self._waiters[dest] += 1
            return lock

    def _release_lock_slot(self, dest: Path) -> None:
        with self._registry_lock:
            self._waiters[dest] -= 1
            if self._waiters[dest] == 0:
                del self._locks[dest]
                del self._waiters[dest]


# One process-wide instance - every PullJob (each request gets its own, see pull_router.py)
# needs to coordinate against the same registry to catch a same-tag race between two requests.
_coordinator = DownloadCoordinator()


def get_download_coordinator() -> DownloadCoordinator:
    return _coordinator
