"""The election and heartbeat timers, changeable while a node runs (they affect liveness, never
safety, so nodes
may briefly disagree while a change spreads).
"""

from dataclasses import replace
from typing import Any

from app.consensus.node import PaxosNode

MIN_ELECTION, MAX_ELECTION = 0.5, 60.0
MIN_HEARTBEAT, MAX_HEARTBEAT = 0.05, 5.0


def valid_timers(value: Any) -> tuple[float, float] | None:
    """(election, heartbeat) seconds from a replicated config value, or None if it makes no sense.
    The election timeout must cover at least three heartbeats, or followers campaign needlessly."""
    try:
        election = float(value["election_seconds"])
        heartbeat = float(value["heartbeat_seconds"])
    except (KeyError, TypeError, ValueError):
        return None
    ok = MIN_ELECTION <= election <= MAX_ELECTION and MIN_HEARTBEAT <= heartbeat <= MAX_HEARTBEAT
    return (election, heartbeat) if ok and heartbeat * 3 <= election else None


def apply_timers(node: PaxosNode, election: float, heartbeat: float) -> None:
    """Gives `node` these timers (the caller holds the node's lock)."""
    node.config = replace(
        node.config, election_timeout=election, heartbeat=heartbeat, retry_interval=election / 2
    )
    node._watch.retune(election)
    node._catchup.retune(node.config.retry_interval / 4)
