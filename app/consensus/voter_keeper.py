"""Keeps the cluster at `target` voters, on the leader, with no outside help (pAIring is only a UI).
A small
cluster has fewer voters: every caught-up learner becomes a voter. A big one keeps `target` and the
rest
follow as learners (more voters only slow decisions). A voter silent for `lost_after` seconds is
replaced by a
caught-up learner and removed. Changing voters needs a quorum, so nothing happens without one.
Target 0 turns it off.
"""

import logging
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING

from app.consensus.probe import is_someone_else

if TYPE_CHECKING:
    from app.consensus.runtime import ConsensusRuntime

logger = logging.getLogger("matricxon.cluster")


class VoterKeeper:
    def __init__(
        self,
        runtime: "ConsensusRuntime",
        target: int,
        interval: float = 5.0,
        lost_after: float = 120.0,
        slack: int = 50,
        stranger: Callable[[str, str], bool] = is_someone_else,
    ) -> None:
        self._stranger = stranger
        self._probed: dict[str, float] = {}
        self._runtime = runtime
        self.target, self._interval, self._lost_after, self._slack = (
            target,
            interval,
            lost_after,
            slack,
        )
        self._missing: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="voter-keeper")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.step()
            except Exception:  # noqa: BLE001 - looked at again next round
                logger.exception("cluster: keeping the voters failed")

    def _stale(self, registered: dict[str, str], heard: dict[str, str], now: float) -> list[str]:
        """Registered nodes whose address now answers as someone else (reset: a new identity), asked
        at most once a
        minute each. A node that doesn't answer stays: it may be down.
        """
        stale = []
        for node_id, address in registered.items():
            if node_id in heard or now - self._probed.get(node_id, -1e9) < 60.0:
                continue
            self._probed[node_id] = now
            if self._stranger(node_id, address):
                stale.append(node_id)
        return stale

    def step(self) -> list[str]:
        """One look; returns what it did (for the log and the tests)."""
        runtime = self._runtime
        with runtime._lock:
            node, now = runtime.node, runtime._clock()
            leading = node.role == "leader" and node.quorum_status(now)[0]
            voters = list(node.members)
            answering = {v for v in voters if v == node.id or node._watch.heard_within(v, now)}
            config = runtime.shared_config()
            removed = set(config.get("cluster/removed", []))
            registered = dict(config.get("cluster/nodes", {}))
            known = runtime.learners.known(now)
            learners = [
                (node_id, address)
                for node_id, address in runtime.learners.caught_up(
                    voters, node.replica.epoch, node.replica.next_slot, self._slack, now
                )
                if node_id not in removed
            ]
        if not leading:
            self._missing.clear()
            return []
        fresh = {
            n: a
            for n, a in known.items()
            if n not in removed and n not in voters and registered.get(n) != a
        }
        gone = self._stale(registered, known, now)
        if fresh or gone:
            nodes = {k: v for k, v in {**registered, **fresh}.items() if k not in gone}
            runtime.submit({"op": "set", "key": "cluster/nodes", "value": nodes})
            for node_id in gone:
                logger.info(
                    "cluster: node %s left the node list (its address answers as another)",
                    node_id[:6],
                )
        if self.target <= 0:
            return []
        self._missing = {v: self._missing.get(v, now) for v in voters if v not in answering}
        lost = [v for v, since in self._missing.items() if now - since >= self._lost_after]
        done: list[str] = []
        for node_id, address in learners[: max(0, self.target - len(voters))]:
            if runtime.add_voter(node_id, address):
                done.append(f"node {node_id[:6]} made a voter")
        if not done and lost and learners and runtime.add_voter(*learners[0]):
            done.append(f"node {learners[0][0][:6]} made a voter, replacing {lost[0][:6]}")
            runtime.remove_voter(lost[0])
        for line in done:
            logger.info("cluster: %s (the cluster keeps %d voters)", line, self.target)
        return done
