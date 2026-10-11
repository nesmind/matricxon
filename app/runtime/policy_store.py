import json
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path


class PolicyStore(ABC):
    """Where the cache policy lives: a local file now, cluster config later."""

    @abstractmethod
    def read(self) -> dict | None:
        """The saved policy, or None if nothing was saved. Raises ValueError/OSError if bad."""

    @abstractmethod
    def write(self, data: dict) -> None: ...


class FilePolicyStore(PolicyStore):
    """One JSON file, replaced atomically so a reader never sees half a write."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text())
        except FileNotFoundError:
            return None

    def write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data))
        os.replace(tmp, self.path)
