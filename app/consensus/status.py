from typing import TYPE_CHECKING, Any

from app.config import settings

if TYPE_CHECKING:
    from app.consensus.runtime import ConsensusRuntime


class RuntimeStatus:
    """What GET /api/consensus/status says about a running node."""

    def __init__(self, runtime: "ConsensusRuntime") -> None:
        self._runtime = runtime

    @staticmethod
    def _nodes(node: Any, runtime: "ConsensusRuntime") -> dict[str, str]:
        """The cluster's own node list: voters plus registered learners, minus any taken out on
        purpose. This, not an
        admin tool's host list, says who is in the cluster.
        """
        config = runtime._config_view()
        removed = set(config.get("cluster/removed", []))
        learners = {k: v for k, v in config.get("cluster/nodes", {}).items() if k not in removed}
        return {**learners, **node.replica.members}

    def report(self) -> dict[str, Any]:
        runtime = self._runtime
        with runtime._lock:
            node = runtime.node
            has_quorum, reachable = node.quorum_status(runtime._clock())
            status = {
                "node_id": node.id,
                "has_quorum": has_quorum,
                "reachable": reachable,
                "epoch": node.epoch,
                "role": node.role,
                "leader": node.leader,
                "is_voter": node.is_voter,
                "voters": node.members,
                "members": dict(node.replica.members),
                "cluster_id": runtime._config_view().get("cluster/id"),
                "applied_slot": node.replica.next_slot,
                "applied_total": node.replica.applied_total(),
                "config": runtime._config_view(),
                "timers": {
                    "election_seconds": node.config.election_timeout,
                    "heartbeat_seconds": node.config.heartbeat,
                },
                "target_voters": runtime.voter_keeper.target,
                "nodes": self._nodes(node, runtime),
                "removed": list(runtime._config_view().get("cluster/removed", [])),
                "note": runtime.note,
                "advertise_url": settings.advertise_url or None,
                "node_kind": settings.node_kind,
                "admin_auth": bool(settings.admin_secret),
                # What the node would use to reach each peer (a voter's own address wins).
                "addresses": {**node.replica.addresses, **runtime._known, **node.replica.members},
                "seeds": dict(runtime._known),
            }
        names = runtime.keeper.names if runtime.keeper else ()
        return {
            **status,
            "leases": runtime.leases.snapshot(),
            "holds": [n for n in names if runtime.leases.holds(n)],
        }
