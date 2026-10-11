"""Hard reset of a node to a fresh install so it can be added again. It checks nothing about the
state (member, removed, standalone, half set up): cluster settings in .env (folders back to
defaults), consensus journal, saved-cache policy, own saved chat caches, node identity and cluster
markers go. Models and shared folders are never touched. One refusal: an active voter must leave the
cluster first (unless `force`, the whole-cluster reset)."""

import json
import threading
from collections.abc import Callable
from pathlib import Path

from app.cluster.provisioning import EnvFile
from app.config import Settings
from app.storage.paths import StoragePaths

# What provisioning wrote into the .env of a joiner.
CLUSTER_ENV_KEYS = [
    "MATRICXON_CLUSTER_ROLE",
    "MATRICXON_MODELS_DIR",
    "MATRICXON_PROMPT_CACHE_DIR",
    "MATRICXON_ADVERTISE_URL",
    "MATRICXON_CONSENSUS_SEEDS",
    "MATRICXON_ADMIN_SECRET",
    "MATRICXON_NODE_KIND",
]


class ResetRefusedError(Exception):
    """This node may not be reset (the message says why)."""


class NodeResetter:
    def __init__(
        self,
        settings: Settings,
        paths: StoragePaths,
        env: EnvFile,
        is_voter: Callable[[], bool],
        restart: Callable[[], None],
        delay: float = 1.5,
        own_cache_dir: Path | None = None,
    ) -> None:
        self._own_cache_dir = own_cache_dir or Path(
            Settings.model_fields["prompt_cache_dir"].default
        )
        self._settings = settings
        self._paths = paths
        self._env = env
        self._is_voter = is_voter
        self._restart = restart
        self._delay = delay

    def reset(self, force: bool = False) -> bool:
        """True when a restart was scheduled (after the HTTP reply). `force` also resets a voter:
        for wiping the whole cluster, with no member left to tell."""
        if self._is_voter() and not force:
            raise ResetRefusedError(
                "this node is still a voter of its cluster: remove it from there first"
            )
        self._clear_markers(force)
        self._env.remove(CLUSTER_ENV_KEYS)
        for path in [
            *self._leftovers(),
            self._paths.node_id_file,
        ]:
            path.unlink(missing_ok=True)
        threading.Timer(self._delay, self._restart).start()
        return True

    def _leftovers(self) -> list[Path]:
        """What a node keeps of its time in a cluster: the journal, saved-cache policy and its own
        saved chat caches (a
        speed-up rebuilt on use; deleted only in the node's default folder, never a shared one).
        """
        journal = self._paths.consensus_journal
        found = [*journal.parent.glob(f"{journal.name}*"), self._paths.cache_policy_file]
        folder = self._paths.prompt_cache_dir
        if folder.is_dir() and folder.resolve() == self._own_cache_dir.resolve():
            found += folder.glob("*.mxc")
        return [path for path in found if path.exists()]

    def _clear_markers(self, force: bool) -> None:
        """Clears this node's cluster markers in the shared folder (not data) and its seed claim. A
        forced (whole-cluster) reset clears all, so no old claim blocks a new first node."""
        markers = self._paths.cluster_dir
        try:
            own = self._paths.node_id_file.read_text().strip()
        except OSError:
            own = ""
        seed = markers / "seed.json"
        try:
            claimed_by = json.loads(seed.read_text()).get("node_id")
        except (OSError, ValueError):
            claimed_by = None
        if force or (own and claimed_by == own):
            seed.unlink(missing_ok=True)
        files = markers.glob("nodes/*.json") if force else [markers / "nodes" / f"{own}.json"]
        for path in files:
            path.unlink(missing_ok=True)
