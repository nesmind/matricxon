import os
from pathlib import Path


class PidFile:
    """Writes this process's PID to `path` on write, removes it on remove. The single source of
    truth scripts/start.sh, stop.sh and status.sh read (via MATRICXON_PID_FILE): self-reported by
    the process (os.getpid), not captured by the launching shell's `$!`, which is only as reliable
    as how the process was spawned (background job, setsid, supervisor, container entrypoint...). A
    self-written file is right regardless."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def previous_run_died(self) -> bool:
        """True if the file names another process that is gone: the last run was killed or crashed.
        A clean stop removes the file; an in-place restart keeps the same pid: neither counts."""
        try:
            previous = int(self._path.read_text().strip())
        except (OSError, ValueError):
            return False
        if previous == os.getpid():
            return False
        try:
            os.kill(previous, 0)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        return False

    def write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(str(os.getpid()))

    def remove(self) -> None:
        self._path.unlink(missing_ok=True)
