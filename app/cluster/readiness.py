import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.cluster.freshness import FreshnessChecker
from app.cluster.fs_probe import FilesystemProbe
from app.config import Settings
from app.storage.paths import StorageInspector, StoragePaths

OK, ACTION = "ok", "action_needed"


@dataclass(frozen=True)
class Check:
    id: str
    status: str
    what: str
    why: str
    setting: str | None = None
    size_bytes: int = 0
    hint: str = ""


@dataclass(frozen=True)
class ReadinessReport:
    role: str
    node_id: str
    checks: list[Check]
    active: bool
    nodes: list[str]
    node_kind: str = "compute"
    admin_auth: bool = False

    @property
    def ready(self) -> bool:
        return all(c.status == OK for c in self.checks)


class ClusterReadiness:
    """What the admin must do to run this node in a cluster, verified (docs/CLUSTER_STORAGE.md
    sections 6-7). A
    cluster is one seed plus fresh joiners; nothing is merged or moved here.
    """

    def __init__(
        self,
        settings: Settings,
        paths: StoragePaths,
        node_id: str,
        probe: FilesystemProbe | None = None,
        inspector: StorageInspector | None = None,
        consensus_running: Callable[[], bool] = lambda: False,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._node_id = node_id
        self._probe = probe or FilesystemProbe()
        self._inspector = inspector or StorageInspector()
        self._freshness = FreshnessChecker(settings, paths)
        self._consensus_running = consensus_running

    @property
    def _joiner(self) -> bool:
        return self._settings.cluster_role == "joiner"

    # report

    def report(self) -> ReadinessReport:
        """Checks only (a role of "off" shows what a seed would need)."""
        checks = [
            self._shared_dir(
                "models", self._paths.models_dir, "MATRICXON_MODELS_DIR", write=not self._joiner
            ),
            self._shared_dir(
                "prompt_cache", self._paths.prompt_cache_dir, "MATRICXON_PROMPT_CACHE_DIR", True
            ),
            self._key(),
        ]
        if not self._consensus_running():
            # First-setup checks (which node is the seed, does a joiner see its shared folder): once
            # the nodes agree they only add noise.
            checks += (
                [self._fresh(), self._seed_visible()] if self._joiner else [self._single_seed()]
            )
        return ReadinessReport(
            self._settings.cluster_role,
            self._node_id,
            checks,
            self._is_active(),
            self._nodes(),
            self._settings.node_kind,
            bool(self._settings.admin_secret),
        )

    def activate(self) -> ReadinessReport:
        """Announces this node in the shared dir, but only when every check passes (a seed also
        claims the cluster by writing seed.json)."""
        current = self.report()
        if self._settings.cluster_role == "off" or not current.ready:
            return current
        if not self._joiner:
            self._write(self._paths.cluster_dir / "seed.json", {"node_id": self._node_id})
        self._write(self._node_file(self._node_id), {"role": self._settings.cluster_role})
        return self.report()

    # checks

    def _shared_dir(self, check_id: str, path: Path, setting: str, write: bool) -> Check:
        size = self._inspector.usage(path).size_bytes
        why = "all nodes must read the same files" + (" and write them" if write else "")
        move = f"Move {path} ({size / 1e9:.2f} GB) to shared storage, mount it on every node"
        if not self._settings.cluster_trust_shared_paths and not self._probe.is_shared(path):
            kind = self._probe.fstype(path) or "unknown"
            hint = f"{move}, then set {setting}. Now on a local '{kind}' filesystem."
            return Check(
                check_id, ACTION, f"{check_id}: not on shared storage", why, setting, size, hint
            )
        if not self._accessible(path, write):
            hint = f"{path} is missing or not {'writable' if write else 'readable'} by this node."
            return Check(check_id, ACTION, f"{check_id}: path not usable", why, setting, size, hint)
        return Check(check_id, OK, f"{check_id}: shared and usable", why, setting, size)

    def _key(self) -> Check:
        why = "every node must seal and open cache files with the same key"
        if self._settings.cache_key or self._paths.cache_key_file.exists():
            return Check("cache_key", OK, "cache_key: provisioned", why, "MATRICXON_CACHE_KEY")
        hint = (
            f"Set the same MATRICXON_CACHE_KEY on every node, or move {self._paths.cache_key_file}."
        )
        return Check("cache_key", ACTION, "cache_key: missing", why, "MATRICXON_CACHE_KEY", 0, hint)

    def _fresh(self) -> Check:
        why = "a joiner must be a fresh install; populated nodes are never merged"
        extra = self._freshness.local_state()
        if not extra:
            return Check("node_fresh", OK, "node_fresh: no local state", why)
        hint = f"{self._settings.node_state_dir} holds {', '.join(extra)}. Use a clean install."
        return Check("node_fresh", ACTION, "node_fresh: node has its own data", why, None, 0, hint)

    def _seed_visible(self) -> Check:
        why = "proves this node mounts the same shared directory as the seed"
        if (self._paths.cluster_dir / "seed.json").is_file():
            return Check("seed_visible", OK, "seed_visible: seed found", why)
        hint = (
            f"No seed marker in {self._paths.cluster_dir}. Mount the seed's dir, activate the seed."
        )
        return Check("seed_visible", ACTION, "seed_visible: seed not found", why, None, 0, hint)

    def _single_seed(self) -> Check:
        why = "one cluster has exactly one seed"
        marker = self._paths.cluster_dir / "seed.json"
        owner = self._read(marker).get("node_id") if marker.is_file() else None
        if owner in (None, self._node_id):
            return Check("single_seed", OK, "single_seed: no other seed", why)
        hint = (
            f"Seed {owner} already claimed this shared dir. Merging populated nodes is unsupported."
        )
        return Check("single_seed", ACTION, "single_seed: another seed exists", why, None, 0, hint)

    # helpers

    def _accessible(self, path: Path, write: bool) -> bool:
        if not path.is_dir():
            return False
        if not write:
            return os.access(path, os.R_OK)
        try:
            fd, tmp = tempfile.mkstemp(dir=path, prefix=f".probe-{self._node_id}-")
        except OSError:
            return False
        os.close(fd)
        Path(tmp).unlink(missing_ok=True)
        return True

    def _node_file(self, node_id: str) -> Path:
        return self._paths.cluster_dir / "nodes" / f"{node_id}.json"

    def _is_active(self) -> bool:
        return self._node_file(self._node_id).is_file()

    def _nodes(self) -> list[str]:
        folder = self._paths.cluster_dir / "nodes"
        return sorted(p.stem for p in folder.glob("*.json")) if folder.is_dir() else []

    @staticmethod
    def _read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _write(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = data | {"at": time.time()}
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(data))
        os.replace(tmp, path)
