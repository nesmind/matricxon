import ctypes
import ctypes.util
import gc
import logging
import threading

logger = logging.getLogger(__name__)


class MemoryTrimmer:
    """Hands memory freed by an unloaded (or just loaded) model back to the system.

    glibc keeps freed memory in per-thread pools: loading runs on worker threads, so an unloaded
    model's ~2.5GB stayed counted against the process (measured: 3177MB idle -> 650MB after
    `malloc_trim(0)`), leaving less RAM for the next load. No-op without glibc (e.g. macOS).
    """

    DELAY_SECONDS = 1.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._malloc_trim = self._find_malloc_trim()

    @staticmethod
    def _find_malloc_trim() -> "ctypes._NamedFuncPointer | None":
        name = ctypes.util.find_library("c")
        if name is None:
            return None
        try:
            return ctypes.CDLL(name).malloc_trim
        except (OSError, AttributeError):
            return None

    def schedule(self) -> None:
        """Trims shortly after, once the caller's last references are gone; repeated calls in
        the window collapse into one."""
        if self._malloc_trim is None:
            return
        with self._lock:
            if self._timer is not None:
                return
            self._timer = threading.Timer(self.DELAY_SECONDS, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self) -> None:
        with self._lock:
            self._timer = None
        gc.collect()
        try:
            self._malloc_trim(0)
        except Exception:  # noqa: BLE001 - trimming is only an optimization
            logger.exception("malloc_trim failed")


memory_trimmer = MemoryTrimmer()
