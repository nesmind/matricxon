"""One plain log line per cluster event as seen from this node: started or stopped, role changed,
leader elected, voters changed, quorum lost or back, another node stopped answering or answers
again, a lease moved.

They use their own always-on logger, so a changing cluster stays readable at log level 0 (warnings
only). Important ones are warnings; the rest are info lines this logger lets through.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from app.consensus.node import PaxosNode

LOGGER_NAME = "matricxon.cluster"
logger = logging.getLogger(LOGGER_NAME)


def short(node_id: str | None) -> str:
    return (node_id or "none")[:6]


def enable_always_on() -> None:
    """Lets this logger's info lines through even when the root level is WARNING (log level 0)."""
    logger.setLevel(logging.INFO)


@dataclass
class _Peer:
    failing_since: float | None = None
    down: bool = False


class ClusterEvents:
    UNREACHABLE_AFTER_SECONDS = 5.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._seen: dict | None = None
        self._peers: dict[str, _Peer] = {}

    # lifecycle

    def started(self, node: PaxosNode) -> None:
        kind = "a voter" if node.is_voter else "a learner (not a voter yet)"
        voters = ", ".join(short(v) for v in node.members)
        logger.info(
            "cluster: node %s started consensus as %s, epoch %d, voters [%s]",
            short(node.id),
            kind,
            node.epoch,
            voters,
        )

    def stopped(self, node: PaxosNode) -> None:
        logger.info("cluster: node %s is stopping (it was %s)", short(node.id), node.role)

    @staticmethod
    def previous_run_crashed() -> None:
        logger.warning(
            "cluster: the previous run of this node did not stop cleanly "
            "(crash, kill or power loss)"
        )

    # what changes while it runs (called every tick, under the runtime's lock)

    def observe(self, node: PaxosNode, config: dict) -> None:
        now = self._clock()
        has_quorum, reachable = node.quorum_status(now)
        state = {
            "role": node.role,
            "leader": node.leader,
            "epoch": node.epoch,
            "voters": set(node.members),
            "quorum": has_quorum,
            "leases": {
                k.removeprefix("lease/"): (v or {}).get("holder")
                for k, v in config.items()
                if k.startswith("lease/")
            },
        }
        before, self._seen = self._seen, state
        if before is None:
            return
        self._roles(node, before, state)
        self._voters(node, before, state)
        self._quorum(node, before, state, reachable)
        for lease, holder in state["leases"].items():
            if holder != before["leases"].get(lease) and holder:
                logger.info("cluster: lease %s is now owned by node %s", lease, short(holder))

    @staticmethod
    def _roles(node: PaxosNode, before: dict, now: dict) -> None:
        if now["role"] != before["role"]:
            if now["role"] == "leader":
                logger.info(
                    "cluster: node %s is now the LEADER (epoch %d)", short(node.id), node.epoch
                )
            elif now["role"] == "candidate":
                logger.info("cluster: node %s is campaigning: no leader heard", short(node.id))
        if now["leader"] != before["leader"] and now["role"] != "leader":
            if now["leader"] is None:
                logger.info("cluster: no leader right now (an election is running)")
            else:
                logger.info("cluster: the leader is now node %s", short(now["leader"]))

    @staticmethod
    def _voters(node: PaxosNode, before: dict, now: dict) -> None:
        joined, left = now["voters"] - before["voters"], before["voters"] - now["voters"]
        if not (joined or left):
            return
        parts = [f"joined: {', '.join(short(v) for v in sorted(joined))}"] if joined else []
        parts += [f"left: {', '.join(short(v) for v in sorted(left))}"] if left else []
        voters = ", ".join(short(v) for v in node.members)
        logger.info(
            "cluster: voters changed (epoch %d): %s; now [%s]", node.epoch, "; ".join(parts), voters
        )

    @staticmethod
    def _quorum(node: PaxosNode, before: dict, now: dict, reachable: int | None) -> None:
        if not node.is_voter or now["quorum"] == before["quorum"]:
            return
        if now["quorum"]:
            logger.info("cluster: quorum is back; cluster changes work again")
        else:
            seen = f" ({reachable} of {len(node.members)} voters reachable)" if reachable else ""
            logger.warning(
                "cluster: NO QUORUM%s; cluster changes are paused, chats keep working", seen
            )

    # other nodes answering or not (called by the sender threads)

    def delivery(self, peer: str, address: str | None, ok: bool) -> None:
        state = self._peers.setdefault(peer, _Peer())
        if ok:
            if state.down:
                logger.info("cluster: node %s (%s) answers again", short(peer), address)
            state.failing_since, state.down = None, False
            return
        now = self._clock()
        state.failing_since = state.failing_since if state.failing_since is not None else now
        if not state.down and now - state.failing_since >= self.UNREACHABLE_AFTER_SECONDS:
            state.down = True
            logger.warning(
                "cluster: node %s (%s) is not answering (down, crashed or cut off)",
                short(peer),
                address,
            )
