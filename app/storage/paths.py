import os
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.config import Settings

_PROJECT_DIR = Path(__file__).resolve().parents[2]


class StorageRole(StrEnum):
    SHARED = "shared"
    REPLICATED = "replicated"
    CLUSTER_CONFIG = "cluster_config"
    PER_NODE = "per_node"


@dataclass(frozen=True)
class StorageEntry:
    name: str
    path: Path
    role: StorageRole
    setting: str | None = None


@dataclass(frozen=True)
class StorageUsage:
    exists: bool
    size_bytes: int
    free_bytes: int | None


class StoragePaths:
    """The one place that resolves every path the node reads or writes (docs/CLUSTER_STORAGE.md)."""

    def __init__(self, settings: Settings, pid_file: Path | None = None) -> None:
        self._settings = settings
        self._pid_file = pid_file or Path(os.environ.get("MATRICXON_PID_FILE", "run/matricxon.pid"))

    @property
    def state_dir(self) -> Path:
        return self._settings.prompt_cache_dir.parent

    @property
    def models_dir(self) -> Path:
        return self._settings.models_dir

    @property
    def prompt_cache_dir(self) -> Path:
        return self._settings.prompt_cache_dir

    @property
    def cache_key_file(self) -> Path:
        return self.state_dir / "prompt_cache.key"

    @property
    def cache_policy_file(self) -> Path:
        return self.state_dir / "prompt_cache_policy.json"

    @property
    def node_id_file(self) -> Path:
        return self._settings.node_state_dir / "node_id"

    @property
    def cluster_dir(self) -> Path:
        """Cluster markers (seed.json, nodes/<id>.json), next to the shared cache dir."""
        return self.state_dir / ".cluster"

    @property
    def consensus_journal(self) -> Path:
        return self._settings.node_state_dir / "consensus.journal"

    @property
    def env_file(self) -> Path:
        return Path(".env")

    @property
    def log_file(self) -> Path:
        return Path("logs/matricxon.log")

    @property
    def pid_file(self) -> Path:
        return self._pid_file

    def entries(self) -> list[StorageEntry]:
        S, R, C, N = (
            StorageRole.SHARED,
            StorageRole.REPLICATED,
            StorageRole.CLUSTER_CONFIG,
            StorageRole.PER_NODE,
        )
        return [
            StorageEntry("models", self.models_dir, S, "MATRICXON_MODELS_DIR"),
            StorageEntry("prompt_cache", self.prompt_cache_dir, S, "MATRICXON_PROMPT_CACHE_DIR"),
            StorageEntry("cache_key", self.cache_key_file, R, "MATRICXON_CACHE_KEY"),
            StorageEntry("cache_policy", self.cache_policy_file, C),
            StorageEntry("env", self.env_file, N),
            StorageEntry("node_id", self.node_id_file, N),
            StorageEntry("consensus_journal", self.consensus_journal, N),
            StorageEntry("native_build", _PROJECT_DIR / "app/native/build", N),
            StorageEntry("numba_cache", _PROJECT_DIR / "app/native/__pycache__", N),
            StorageEntry("log", self.log_file, N),
            StorageEntry("pid", self.pid_file, N, "MATRICXON_PID_FILE"),
        ]


class StorageInspector:
    """Size and free space of a path (free space of the nearest existing ancestor's disk)."""

    def usage(self, path: Path) -> StorageUsage:
        exists = path.exists()
        return StorageUsage(exists, self._size(path) if exists else 0, self._free(path))

    @staticmethod
    def _size(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        total = 0
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    pass
        return total

    @staticmethod
    def _free(path: Path) -> int | None:
        probe = path.resolve()
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free
        except OSError:
            return None
