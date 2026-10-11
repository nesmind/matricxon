from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.storage.paths import StoragePaths


@dataclass(frozen=True)
class FreshnessReport:
    fresh: bool
    reasons: list[str]


class FreshnessChecker:
    """Is this a fresh install? Only a fresh node may join a cluster (docs/CLUSTER_STORAGE.md
    section 7); populated
    nodes are never merged. A cache key alone doesn't count: a node makes one at first start, and a
    joiner uses the shared key.
    """

    def __init__(self, settings: Settings, paths: StoragePaths) -> None:
        self._settings = settings
        self._paths = paths

    def report(self) -> FreshnessReport:
        reasons: list[str] = []
        if self._settings.cluster_role != "off":
            reasons.append(f"already has the cluster role '{self._settings.cluster_role}'")
        models = self._count(self._paths.models_dir, "*.gguf.json")
        if models:
            reasons.append(f"{models} installed model(s) in its own models folder")
        chats = self._count(self._paths.prompt_cache_dir, "*.mxc")
        if chats:
            reasons.append(f"{chats} saved chat cache file(s)")
        if self._paths.cache_policy_file.exists():
            reasons.append("its own saved cache policy")
        extra = self.local_state(include_journal=True)
        if extra:
            reasons.append(f"other local state: {', '.join(extra)}")
        return FreshnessReport(not reasons, reasons)

    @staticmethod
    def _count(folder: Path, pattern: str) -> int:
        return sum(1 for _ in folder.rglob(pattern)) if folder.is_dir() else 0

    def local_state(self, include_journal: bool = False) -> list[str]:
        """Names in the node state folder that mean it has a history (not a first boot's)."""
        state = self._settings.node_state_dir
        if not state.is_dir():
            return []
        ignored = {
            "node_id",
            "oracle",
            "models",
            "prompt_cache",
            "prompt_cache.key",
            "prompt_cache_policy.json",
        }
        if not include_journal:  # a node already in the cluster naturally has its consensus journal
            ignored.add("consensus.journal")
        return sorted(p.name for p in state.iterdir() if p.name not in ignored)
