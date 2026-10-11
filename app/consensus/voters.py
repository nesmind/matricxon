"""Changing who votes, and where the voters can be reached (each change is a new epoch)."""

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.consensus.runtime import ConsensusRuntime

logger = logging.getLogger(__name__)


class VoterChanges:
    def __init__(self, runtime: "ConsensusRuntime") -> None:
        self._runtime = runtime

    # Replicated config: ids taken out on purpose (never re-added by the voter keeper) and the
    # learners the cluster knows (id -> address; voters are in the members).
    REMOVED_KEY = "cluster/removed"
    NODES_KEY = "cluster/nodes"

    def add(self, node_id: str, address: str, wait: float) -> bool:
        ok = self._add(node_id, address, wait)
        removed = self._removed()
        if ok and node_id in removed:
            self._store_removed([n for n in removed if n != node_id])
        return ok

    def _removed(self) -> list[str]:
        return list(self._runtime.shared_config().get(self.REMOVED_KEY, []))

    def _store_removed(self, ids: list[str]) -> None:
        self._runtime.submit({"op": "set", "key": self.REMOVED_KEY, "value": ids[-100:]})

    def _add(self, node_id: str, address: str, wait: float) -> bool:
        """Makes `node_id` a voter (a new epoch). True once it is one; False if it didn't work out
        in time (it may still apply later) or the address doesn't answer as that node (an
        unreachable voter would cost the quorum).

        A node already a voter under another address gets the new one recorded with the same voters.
        The voter-list address is what other nodes use and is never re-read from settings, so a
        moved node (new IP) would stay unreachable under the old one.
        """
        probe = self._runtime._probe
        if probe is not None and not probe(node_id, address):
            logger.warning(
                "cluster: not making node %s a voter: %s does not answer as it",
                node_id[:6],
                address,
            )
            return False
        recorded = self._members().get(node_id)
        if recorded is not None and recorded != address:
            return self._change(
                lambda m: {**m, node_id: address}, lambda m: m.get(node_id) == address, wait
            )
        return self._change(lambda m: {**m, node_id: address}, lambda m: node_id in m, wait)

    def remove(self, node_id: str, wait: float, leave: bool = False) -> bool:
        """Takes `node_id` out of the voters (a new epoch). With `leave` it is remembered as removed
        on purpose, so it isn't
        promoted while it still follows the log. Written first: the node asked may be the one going
        out, and then nobody hears its writes.
        """
        marked = self._mark_left(node_id) if leave else None
        ok = self._change(
            lambda m: {k: v for k, v in m.items() if k != node_id},
            lambda m: node_id not in m,
            wait,
            keep_one=True,
        )
        if marked is not None and not ok:
            self._store_removed(marked)
        return ok

    def _mark_left(self, node_id: str) -> list[str]:
        """Marks `node_id` removed on purpose and off the node list; returns the marks before."""
        before = self._removed()
        if node_id not in before:
            self._store_removed([*before, node_id])
        nodes = self._runtime.shared_config().get(self.NODES_KEY, {})
        if node_id in nodes:
            kept = {k: v for k, v in nodes.items() if k != node_id}
            self._runtime.submit({"op": "set", "key": self.NODES_KEY, "value": kept})
        return before

    def _members(self) -> dict[str, str]:
        with self._runtime._lock:
            return dict(self._runtime.node.replica.members)

    def _change(
        self,
        change: Callable[[dict], dict],
        done: Callable[[dict], bool],
        wait: float,
        keep_one: bool = False,
    ) -> bool:
        runtime = self._runtime
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            with runtime._lock:
                members = dict(runtime.node.replica.members)
                if done(members):
                    return True
                if not members or (keep_one and len(members) <= 1):
                    return False
                command = runtime.node.reconfigure(change(members), runtime._clock())
            while time.monotonic() < deadline:
                with runtime._lock:
                    if runtime.node.replica.applied(command.cmd_id):
                        break
                time.sleep(0.02)
        return done(self._members())
