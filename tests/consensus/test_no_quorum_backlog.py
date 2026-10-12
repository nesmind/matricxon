"""Without a quorum nothing may pile up: abandoned commands leave the retry list, and lease
renewals are not submitted (they used to fill the journal with slots never chosen)."""

import random
import time

from app.consensus.auth import MessageSigner
from app.consensus.config import PaxosConfig
from app.consensus.lease import LeaseKeeper
from app.consensus.node import PaxosNode
from app.consensus.pending import PendingCommands
from app.consensus.runtime import ConsensusRuntime
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import MemoryStore
from tests.consensus.test_leases import Cluster
from tests.consensus.test_wire import Hub


def _runtime(genesis: dict[str, str], **kwargs) -> tuple[Hub, ConsensusRuntime]:
    hub = Hub()
    config = PaxosConfig(heartbeat=0.03, election_timeout=0.15, retry_interval=0.1)
    node = PaxosNode(
        "a",
        MemoryStore(),
        ClusterStateMachine(),
        lambda to, msg: None,
        random.Random("a"),
        config,
        genesis=genesis,
    )
    runtime = ConsensusRuntime(node, MessageSigner(b"s"), genesis, hub.post, **kwargs)
    hub.runtimes["a"] = runtime
    return hub, runtime


def _spy_on_submits(runtime: ConsensusRuntime) -> list[object]:
    calls: list[object] = []
    original = runtime.leases._submit

    def spy(payload, wait):
        calls.append(payload)
        return original(payload, wait)

    runtime.leases._submit = spy
    return calls


def test_pending_forgets_a_discarded_command():
    node = PaxosNode(
        "a", MemoryStore(), ClusterStateMachine(), lambda to, msg: None, random.Random(1)
    )
    pending = PendingCommands(node.replica)
    command = node.new_command({"op": "set", "key": "k", "value": 1})
    pending.add(command)
    pending.discard(command.cmd_id)
    pending.discard("never-added")
    assert pending.open() == []


def test_a_submit_that_times_out_is_not_retried_forever():
    _, runtime = _runtime({"a": "http://a", "b": "http://b"})  # "b" never answers
    runtime.start()
    try:
        for i in range(5):
            _, applied, _ = runtime.submit({"op": "set", "key": "k", "value": i}, wait=0.05)
            assert not applied
        assert runtime.node._pending.open() == []
    finally:
        runtime.stop()


def test_the_lease_keeper_stays_quiet_while_there_is_no_quorum():
    cluster = Cluster()
    ready = [False]
    keeper = LeaseKeeper(cluster.leases["a"], ("pull", "clean"), ttl=9, ready=lambda: ready[0])
    keeper.tick()
    assert not keeper.holds("pull") and cluster.machine.config == {}
    ready[0] = True
    keeper.tick()
    assert keeper.holds("pull") and keeper.holds("clean")


def test_a_node_without_quorum_submits_no_lease_renewals():
    _, runtime = _runtime({"a": "http://a", "b": "http://b"}, lease_names=("pull",), lease_ttl=0.3)
    calls = _spy_on_submits(runtime)
    runtime.start()
    try:
        time.sleep(0.8)  # several keeper rounds
        assert calls == []
        assert runtime.node._pending.open() == []
    finally:
        runtime.stop()


def test_a_node_with_quorum_still_renews_its_leases():
    _, runtime = _runtime({"a": "http://a"}, lease_names=("pull",), lease_ttl=0.3)
    calls = _spy_on_submits(runtime)
    runtime.start()
    try:
        deadline = time.monotonic() + 5
        while not calls and time.monotonic() < deadline:
            time.sleep(0.02)
        assert calls
    finally:
        runtime.stop()
