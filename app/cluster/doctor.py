"""What is wrong with a cluster, from what the nodes report, and how to fix it. Pure analysis; the
reports are
fetched and fixes applied by scripts/cluster_doctor.sh.

Actions a finding can carry: `drop-voter VIA ID`, `add-voter VIA ID ADDRESS`, `recover URL NODE_ID`,
`reset-node URL`; empty when only a person can fix it.
"""

from dataclasses import dataclass
from urllib.parse import urlparse

SLACK = 50  # slots behind that still count as caught up (as the leader's voter keeper)
_LOCAL = {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    node: str
    text: str
    action: str = ""

    def line(self) -> str:
        return "\t".join([self.severity, self.code, self.node, self.text, self.action])


@dataclass(frozen=True)
class Probe:
    url: str
    status: dict | None  # /api/consensus/status (None: no answer; enabled False: consensus off)
    ready: dict | None

    @property
    def up(self) -> bool:
        return self.status is not None or self.ready is not None

    @property
    def running(self) -> bool:
        return bool(self.status and self.status.get("enabled"))

    @property
    def node_id(self) -> str:
        return (self.status or {}).get("node_id") or (self.ready or {}).get("node_id") or ""

    def total(self) -> int:
        return (self.status or {}).get("applied_total", 0)


def same_node(a: str, b: str) -> bool:
    """Same scheme, port and machine (localhost, 127.0.0.1 and:1 are one machine)."""
    pa, pb = urlparse(a.rstrip("/")), urlparse(b.rstrip("/"))
    host = lambda p: "local" if p.hostname in _LOCAL else p.hostname  # noqa: E731
    return (pa.scheme, host(pa), pa.port) == (pb.scheme, host(pb), pb.port)


class ClusterDoctor:
    def __init__(self, probes: list[Probe], previous: dict[str, int] | None = None) -> None:
        self.probes = probes
        self.previous = previous or {}  # url -> applied_total a moment ago: stuck or slow?
        self.running = [p for p in probes if p.running]
        self.ref = max(self.running, key=lambda p: (p.status["epoch"], p.total()), default=None)
        self.leader = next((p for p in self.running if p.status["role"] == "leader"), self.ref)

    def findings(self) -> list[Finding]:
        if not any(p.up for p in self.probes):
            return [
                Finding("error", "ALL_DOWN", "-", "no node answers: start them (scripts/start.sh)")
            ]
        out = [
            Finding(
                "error",
                "DOWN",
                p.url,
                "no answer (stopped, crashed, cut off, or not reachable from here at this address)",
            )
            for p in self.probes
            if not p.up
        ]
        if self.ref is None:
            return out + [
                Finding(
                    "info",
                    "NO_CONSENSUS",
                    "-",
                    "no node runs consensus: cluster mode is off or never started",
                )
            ]
        return out + self._identity() + self._voters() + self._learners() + self._nodes()

    def _identity(self) -> list[Finding]:
        ids = {p.status["cluster_id"] for p in self.running if p.status.get("cluster_id")}
        if len(ids) > 1:
            return [
                Finding(
                    "error",
                    "TWO_CLUSTERS",
                    "-",
                    f"nodes belong to {len(ids)} different clusters: reset one side",
                )
            ]
        return (
            []
            if ids
            else [
                Finding(
                    "warn",
                    "NO_ID",
                    "-",
                    "no cluster id yet (no leader with a majority, or an old Matricxon)",
                )
            ]
        )

    def _voters(self) -> list[Finding]:
        ref, out = self.ref.status, []
        voters, members = ref["voters"], ref.get("members", {})
        by_id = {p.node_id: p for p in self.probes if p.node_id}
        live = [v for v in voters if v in by_id and by_id[v].running]
        if len(live) < len(voters) // 2 + 1:
            return [self._lost_quorum(live, voters)]
        if not self.leader.status.get("has_quorum"):
            out.append(
                Finding(
                    "warn",
                    "NO_LEADER",
                    "-",
                    "a majority is up but there is no leader yet: wait a few seconds",
                )
            )
        via = self.leader.url
        for voter in (v for v in voters if v not in live):
            down = next(
                (
                    p.url
                    for p in self.probes
                    if not p.up and same_node(p.url, members.get(voter, "x://"))
                ),
                None,
            )
            text = f"voter {voter[:8]} ({members.get(voter)}) does not answer; a majority is left"
            out.append(
                Finding(
                    "warn",
                    "VOTER_DOWN" if down else "GHOST_VOTER",
                    down or voter[:8],
                    text,
                    f"drop-voter {via} {voter}",
                )
            )
        for p in (p for p in self.running if p.node_id in voters):
            ours, theirs = (
                (p.status.get("advertise_url") or "").rstrip("/"),
                members.get(p.node_id, "").rstrip("/"),
            )
            if ours and theirs and ours != theirs:
                out.append(
                    Finding(
                        "warn",
                        "ADDRESS_MOVED",
                        p.url,
                        f"advertises {ours}, the cluster uses {theirs}",
                        f"add-voter {via} {p.node_id} {ours}",
                    )
                )
        return out

    def _lost_quorum(self, live: list[str], voters: list[str]) -> Finding:
        survivors = [p for p in self.running if p.node_id in live]
        if not survivors:
            text = f"none of the {len(voters)} voters answers: bring one back, or start anew"
            return Finding("error", "ALL_VOTERS_LOST", "-", text)
        members = self.ref.status.get("members", {})
        unknown = [
            v
            for v in voters
            if v not in live
            and not any(same_node(p.url, members.get(v, "x://")) for p in self.probes)
        ]
        if unknown:  # never force a group of one while a voter may not have been asked
            text = f"{len(unknown)} of {len(voters)} voters were not given: pass all URLs"
            return Finding("warn", "UNPROBED_VOTERS", "-", text)
        best = max(survivors, key=lambda p: (p.status["epoch"], p.total()))
        text = (
            f"no majority ({len(live)} of {len(voters)} voters answer): force a group of one here"
        )
        return Finding("error", "NO_QUORUM", best.url, text, f"recover {best.url} {best.node_id}")

    def _learners(self) -> list[Finding]:
        ref, out = self.ref.status, []
        voters, nodes, removed = (
            set(ref["voters"]),
            ref.get("nodes", {}),
            set(ref.get("removed", [])),
        )
        for p in (p for p in self.running if p.node_id not in voters):
            behind, reset = self.ref.total() - p.total(), f"reset-node {p.url}"
            if p.node_id in removed:
                out.append(
                    Finding(
                        "warn",
                        "REMOVED_BUT_RUNNING",
                        p.url,
                        "was taken out on purpose but still runs its old journal",
                        reset,
                    )
                )
            elif behind > SLACK:
                stuck = self.previous.get(p.url) == p.total()
                text = f"replaying the log: {p.total() * 100 // max(self.ref.total(), 1)}%" + (
                    ", not moving" if stuck else ""
                )
                out.append(
                    Finding("warn", "STUCK_LEARNER", p.url, text, reset)
                    if stuck
                    else Finding("info", "REPLAYING", p.url, text)
                )
            elif 0 < len(voters) < ref.get("target_voters", 0) and p.node_id in nodes:
                text = (
                    f"caught up, but the cluster has {len(voters)} of {ref['target_voters']} voters"
                )
                out.append(
                    Finding(
                        "warn",
                        "UNDER_VOTERS",
                        p.url,
                        text,
                        f"add-voter {self.leader.url} {p.node_id} {nodes[p.node_id]}",
                    )
                )
        return out

    def _nodes(self) -> list[Finding]:
        out = []
        for p in self.probes:
            if p.up and not p.running:
                out.append(
                    Finding(
                        "info",
                        "NOT_IN_CLUSTER",
                        p.url,
                        "runs no consensus: add it from pAIring (Add nodes); reset it if not fresh",
                    )
                )
            bad = [c["id"] for c in (p.ready or {}).get("checks", []) if c.get("status") != "ok"]
            if bad and p.running:
                out.append(
                    Finding(
                        "warn",
                        "SHARED_STORAGE",
                        p.url,
                        f"readiness checks failing: {', '.join(bad)}",
                    )
                )
            if p.status and p.status.get("note"):
                out.append(Finding("info", "NOTE", p.url, str(p.status["note"])))
        return out
