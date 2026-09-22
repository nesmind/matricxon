import os
from pathlib import Path


class PidFile:
    """Writes this running process's own PID to `path` on write(), and removes it on remove().

    The single source of truth scripts/start.sh, scripts/stop.sh and scripts/status.sh all read
    (via MATRICXON_PID_FILE) — deliberately self-reported by the process itself (os.getpid()),
    not captured by the launching shell's own `$!`, since that capture is only ever as reliable
    as exactly how the process was spawned (a plain background job, a session leader via setsid,
    a process supervisor, a container entrypoint, ...). A self-written PID file is correct
    regardless of any of that.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(str(os.getpid()))

    def remove(self) -> None:
        self._path.unlink(missing_ok=True)
