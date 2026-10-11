import threading
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class DownloadCoordinator:
    """Serializes concurrent `/api/pull` requests for the same destination file. Without it, two
    in-flight pulls of one tag (e.g. a retry while an orphaned attempt still ran) both streamed to
    `<dest>.partial`; the first to finish renamed it to `dest` and the other crashed the ASGI
    response with a raw `FileNotFoundError` on `partial_path.rename(dest)`, which an SSE client sees
    as a mid-stream connection drop, not an in-band `{"error":...}` line.

    One `threading.Lock` per distinct `dest` (not one global lock: different tags still download in
    parallel), created lazily and dropped when idle.
    """

    def __init__(self) -> None:
        self._registry_lock = threading.Lock()
        self._locks: dict[Path, threading.Lock] = {}
        self._waiters: dict[Path, int] = defaultdict(int)

    @contextmanager
    def guard(self, dest: Path) -> Iterator[None]:
        """Blocks until no other caller is inside `guard(dest)` for this `dest`, then holds it until
        the `with` block exits (success or exception); the caller decides what "already handled by
        someone else" means (PullJob._run checks `dest.exists` first)."""
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


# One process-wide instance: every PullJob (one per request, pull_router.py) must coordinate against
# the same registry to catch a same-tag race between requests.
_coordinator = DownloadCoordinator()


def get_download_coordinator() -> DownloadCoordinator:
    return _coordinator
