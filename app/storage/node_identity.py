import uuid
from pathlib import Path


class NodeIdentity:
    """This node's stable id: created once into `path`, then read back on every start."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> str:
        try:
            value = self._path.read_text().strip()
            if value:
                return value
        except FileNotFoundError:
            pass
        value = uuid.uuid4().hex
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(value)
        return value
