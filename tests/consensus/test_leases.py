"""Leases on top of the cluster's compare-and-set: one owner at a time, takeover when it dies."""

import random

from app.consensus.auth import MessageSigner
from app.consensus.config import PaxosConfig
from app.consensus.lease import LeaseKeeper, Leases
from app.consensus.node import PaxosNode
from app.consensus.runtime import ConsensusRuntime
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import MemoryStore
from tests.consensus.test_wire import Hub, _wait


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Cluster:
    """Several nodes' Leases sharing one state machine, as the replicated log would (in order)."""

    def __init__(self, names=("a", "b", "c")):
        self.machine = ClusterStateMachine()
        self.clock = Clock()
        self.leases = {n: self._leases(n) for n in names}

    def _leases(self, name: str) -> Leases:
        return Leases(name, self._submit, lambda: dict(self.machine.config), self.clock)

    def _submit(self, payload, wait):
        return None, True, self.machine.apply(payload)


def test_a_free_lease_goes_to_the_first_to_ask_and_is_refused_to_the_others():
    cluster = Cluster()
    assert cluster.leases["a"].try_acquire("pull", 10)
    assert not cluster.leases["b"].try_acquire("pull", 10)
    holders = {n: lease.holder("pull") for n, lease in cluster.leases.items()}
    assert holders == {"a": "a", "b": "a", "c": "a"}
    assert cluster.leases["a"].holds("pull") and not cluster.leases["b"].holds("pull")


def test_the_owner_renews_and_another_node_takes_over_only_after_expiry():
    cluster = Cluster()
    cluster.leases["a"].try_acquire("pull", 10)
    cluster.clock.now += 8
    assert cluster.leases["a"].try_acquire("pull", 10)
    cluster.clock.now += 8
    assert not cluster.leases["b"].try_acquire("pull", 10)
    cluster.clock.now += 3
    assert cluster.leases["b"].holder("pull") is None
    assert cluster.leases["b"].try_acquire("pull", 10)
    assert cluster.leases["c"].holder("pull") == "b" and not cluster.leases["a"].holds("pull")


def test_two_nodes_that_both_saw_it_free_cannot_both_win():
    """The compare-and-set is what makes it exclusive: the second one's `expected` is stale."""
    cluster = Cluster()
    seen_free = cluster.machine.config.get("lease/pull")

    def claim(holder: str) -> bool:
        value = {"holder": holder, "expires": 1010}
        return cluster.machine.apply(
            {"op": "cas", "key": "lease/pull", "expected": seen_free, "value": value}
        )

    assert (claim("a"), claim("b")) == (True, False)
    assert cluster.leases["b"].holder("pull") == "a"


def test_release_frees_it_at_once_but_only_for_the_owner():
    cluster = Cluster()
    cluster.leases["a"].try_acquire("clean", 10)
    assert not cluster.leases["b"].release("clean")
    assert cluster.leases["a"].release("clean")
    assert cluster.leases["b"].try_acquire("clean", 10)


def test_holds_needs_a_safety_margin_and_the_snapshot_lists_live_leases_only():
    cluster = Cluster()
    cluster.leases["a"].try_acquire("pull", 9)
    cluster.leases["b"].try_acquire("clean", 9)
    cluster.clock.now += 7
    assert cluster.leases["a"].holds("pull") and not cluster.leases["a"].holds("pull", margin=3)
    assert set(cluster.leases["c"].snapshot()) == {"pull", "clean"}
    cluster.clock.now += 3
    assert cluster.leases["c"].snapshot() == {}


def test_the_keeper_claims_free_leases_keeps_renewing_and_gives_way_to_a_live_owner():
    cluster = Cluster()
    keepers = {
        n: LeaseKeeper(lease, ("pull", "clean"), ttl=9) for n, lease in cluster.leases.items()
    }
    keepers["a"].tick()
    keepers["b"].tick()
    assert keepers["a"].holds("pull") and keepers["a"].holds("clean")
    assert not keepers["b"].holds("pull")
    for _ in range(5):
        cluster.clock.now += 3
        keepers["a"].tick()
        keepers["b"].tick()
    assert keepers["a"].holds("pull") and not keepers["b"].holds("clean")
    cluster.clock.now += 20
    keepers["b"].tick()
    assert keepers["b"].holds("pull") and keepers["b"].holds("clean")


def _runtime(hub: Hub, name: str, genesis: dict[str, str], ttl: float) -> ConsensusRuntime:
    config = PaxosConfig(heartbeat=0.03, election_timeout=0.15, retry_interval=0.1)
    node = PaxosNode(
        name,
        MemoryStore(),
        ClusterStateMachine(),
        lambda to, msg: None,
        random.Random(name),
        config,
        genesis=genesis,
    )
    runtime = ConsensusRuntime(
        node, MessageSigner(b"s"), genesis, hub.post, lease_names=("pull", "clean"), lease_ttl=ttl
    )
    hub.runtimes[name] = runtime
    return runtime


def _start(names: tuple[str, ...], ttl: float) -> tuple[Hub, dict[str, ConsensusRuntime]]:
    hub = Hub()
    genesis = {n: f"http://{n}" for n in names}
    runtimes = {n: _runtime(hub, n, genesis, ttl) for n in names}
    for runtime in runtimes.values():
        runtime.start()
    return hub, runtimes


def _held(runtimes: dict[str, ConsensusRuntime], alive: list[str]) -> list[str]:
    return sorted(lease for n in alive for lease in runtimes[n].status()["holds"])


def test_three_real_nodes_agree_on_one_owner_per_lease_and_a_survivor_takes_over():
    hub, runtimes = _start(("a", "b", "c"), ttl=1.5)
    try:
        assert _wait(lambda: _held(runtimes, list(runtimes)) == ["clean", "pull"], timeout=10)
        owner = next(n for n in runtimes if "pull" in runtimes[n].status()["holds"])
        hub.down.add(owner)
        runtimes[owner].stop()
        survivors = [n for n in runtimes if n != owner]
        assert _wait(lambda: _held(runtimes, survivors) == ["clean", "pull"], timeout=15)
    finally:
        for runtime in runtimes.values():
            runtime.stop()


def test_voters_can_be_added_and_removed_through_a_running_runtime():
    _, runtimes = _start(("a", "b", "c"), ttl=30)
    try:
        assert _wait(lambda: any(r.status()["role"] == "leader" for r in runtimes.values()))
        assert runtimes["a"].remove_voter("c", wait=8)
        assert _wait(lambda: runtimes["b"].status()["voters"] == ["a", "b"])
        assert runtimes["a"].remove_voter("b", wait=8)
        assert _wait(lambda: runtimes["a"].status()["voters"] == ["a"])
        assert runtimes["a"].remove_voter("a", wait=1) is False
        assert runtimes["a"].status()["voters"] == ["a"]

        assert runtimes["a"].add_voter("b", "http://b", wait=8)
        assert runtimes["a"].add_voter("c", "http://c", wait=8)
        assert _wait(lambda: runtimes["c"].status()["voters"] == ["a", "b", "c"])
        assert runtimes["c"].status()["epoch"] == runtimes["a"].status()["epoch"] == 4
        assert runtimes["b"].add_voter("c", "http://c", wait=1)
    finally:
        for runtime in runtimes.values():
            runtime.stop()
