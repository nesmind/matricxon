"""Cluster-wide settings in the replicated config, so every node picks them up itself.

Shared folders (`cluster/paths`) are applied by writing the node's.env and restarting after a
random delay, so
nodes don't all go down together. Consensus timers (`cluster/timers`) and saved-chat-cache limits
(`cluster/cache`) apply live.
"""

import logging
import random
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.cluster.provisioning import EnvFile
from app.config import Settings
from app.consensus.runtime import ConsensusRuntime
from app.consensus.timers import valid_timers
from app.runtime.cache_store import PersistentCacheStore

logger = logging.getLogger("matricxon.cluster")
PATHS_KEY = "cluster/paths"
TIMERS_KEY = "cluster/timers"
CACHE_KEY = "cluster/cache"


def valid_cache(value: Any) -> tuple[bool, int, int] | None:
    """(enabled, budget_mb, ttl_hours) from a replicated config value, or None."""
    try:
        enabled, budget, ttl = value["enabled"], int(value["budget_mb"]), int(value["ttl_hours"])
    except (KeyError, TypeError, ValueError):
        return None
    return (
        (enabled, budget, ttl) if isinstance(enabled, bool) and budget >= 0 and ttl >= 1 else None
    )


class SharedSettings:
    def __init__(
        self,
        runtime: ConsensusRuntime,
        settings: Settings,
        env: EnvFile,
        restart: Callable[[], None],
        cache: Callable[[], PersistentCacheStore] | None = None,
        interval: float = 3.0,
        delay: tuple[float, float] = (5.0, 25.0),
    ) -> None:
        self._runtime = runtime
        self._settings = settings
        self._env = env
        self._restart = restart
        self._cache = cache
        self._interval = interval
        self._delay = delay
        self._restarting = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="cluster-settings")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - tried again next round
                continue

    def tick(self) -> None:
        config = self._runtime.shared_config()
        self._apply_timers(config.get(TIMERS_KEY))
        self._apply_paths(config.get(PATHS_KEY))
        self._apply_cache(config.get(CACHE_KEY))

    def _apply_cache(self, value: Any) -> None:
        wanted = valid_cache(value)
        if wanted is None or self._cache is None:
            return
        store = self._cache()
        policy = store.policy
        if wanted == (policy.enabled, policy.budget_mb, policy.ttl_hours):
            return
        policy.update(*wanted)
        logger.info(
            "cluster: saved-chat-cache limits changed: %s, %d MB, keep %d h",
            "on" if wanted[0] else "off (saved caches deleted)",
            wanted[1],
            wanted[2],
        )
        if not policy.enabled:
            store.clear()
        store.enforce_limits()

    def _apply_timers(self, value: Any) -> None:
        timers = valid_timers(value)
        if timers is None:
            return
        node = self._runtime.node.config
        if (node.election_timeout, node.heartbeat) != timers:
            self._runtime.retune(*timers)
            logger.info(
                "cluster: consensus timers changed to election %.2fs, heartbeat %.2fs", *timers
            )

    def _apply_paths(self, value: Any) -> None:
        if not isinstance(value, dict) or self._restarting:
            return
        wanted = {
            "MATRICXON_MODELS_DIR": value.get("models_dir"),
            "MATRICXON_PROMPT_CACHE_DIR": value.get("prompt_cache_dir"),
        }
        current = {
            "MATRICXON_MODELS_DIR": str(self._settings.models_dir),
            "MATRICXON_PROMPT_CACHE_DIR": str(self._settings.prompt_cache_dir),
        }
        changes = {k: str(Path(v)) for k, v in wanted.items() if v and str(Path(v)) != current[k]}
        if not changes:
            self._runtime.note = None
            return
        missing = [v for v in changes.values() if not Path(v).is_absolute() or not Path(v).is_dir()]
        if missing:  # not mounted here (yet): leave the node, retry next round
            note = f"the cluster folder {missing[0]} is not available on this node"
            if note != self._runtime.note:
                logger.warning("cluster: %s; keeping the current folders", note)
            self._runtime.note = note
            return
        self._env.set(changes)
        self._restarting = True
        self._runtime.note = "restarting to use the cluster's shared folders"
        wait = random.uniform(*self._delay)
        logger.info("cluster: shared folders changed to %s; restarting in %.0f s", changes, wait)
        threading.Timer(wait, self._restart).start()
