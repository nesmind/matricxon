"""What the cluster tells the log, at log level 0 too."""

import logging

import pytest

from app.consensus.events import LOGGER_NAME, ClusterEvents, enable_always_on, short
from tests.consensus.sim import SimCluster


@pytest.fixture
def lines(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    return lambda: [r.getMessage() for r in caplog.records if r.name == LOGGER_NAME]


def _watch(cluster: SimCluster, node_id: str, events: ClusterEvents, seconds: float) -> None:
    """Runs the simulation, letting `events` look at one node after every step like the runtime."""
    end = cluster.now + seconds
    while cluster.now < end:
        cluster.run(0.05)
        node = cluster.nodes.get(node_id)
        if node is not None:
            events.observe(node, dict(node.replica.machine.config))


def test_an_election_a_leader_and_a_lost_quorum_are_all_told(lines):
    cluster = SimCluster(seed=3)
    events = ClusterEvents(clock=lambda: cluster.now)
    cluster.run(2.0)
    _watch(cluster, "n0", events, 0.2)
    cluster.crash("n1")
    cluster.crash("n2")
    _watch(cluster, "n0", events, 6.0)

    text = "\n".join(lines())
    assert "NO QUORUM" in text and "changes are paused, chats keep working" in text


def test_a_leader_change_and_new_voters_are_told(lines):
    cluster = SimCluster(seed=4)
    events = ClusterEvents(clock=lambda: cluster.now)
    cluster.run(2.0)
    _watch(cluster, "n0", events, 0.2)
    cluster.add_node("n3")
    cluster.run(2.0)
    cluster.reconfigure(["n0", "n1", "n2", "n3"])
    _watch(cluster, "n0", events, 5.0)

    text = "\n".join(lines())
    assert f"joined: {short('n3')}" in text and "voters changed" in text


def test_a_node_that_stops_answering_is_reported_once_and_again_when_it_is_back(lines):
    now = [0.0]
    events = ClusterEvents(clock=lambda: now[0])
    for second in range(0, 12):
        now[0] = float(second)
        events.delivery("f873e279", "http://10.0.0.7:8420", ok=False)
    assert sum("is not answering" in line for line in lines()) == 1
    events.delivery("f873e279", "http://10.0.0.7:8420", ok=True)
    events.delivery("f873e279", "http://10.0.0.7:8420", ok=True)
    assert sum("answers again" in line for line in lines()) == 1


def test_a_short_blip_is_not_reported(lines):
    now = [0.0]
    events = ClusterEvents(clock=lambda: now[0])
    events.delivery("peer", "http://x", ok=False)
    now[0] = 2.0
    events.delivery("peer", "http://x", ok=False)
    events.delivery("peer", "http://x", ok=True)
    assert lines() == []


def test_the_cluster_logger_speaks_even_when_everything_else_is_at_warning():
    """Log level 0: the root logger at WARNING and a handler that takes everything (basicConfig)."""
    seen: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    root, cluster_logger = logging.getLogger(), logging.getLogger(LOGGER_NAME)
    handler, old_levels = Collect(), (root.level, cluster_logger.level)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    cluster_logger.setLevel(logging.NOTSET)
    try:
        logging.getLogger("matricxon.other").info("not cluster news")
        assert seen == []
        enable_always_on()
        cluster_logger.info("cluster: test line")
        assert seen == ["cluster: test line"]
    finally:
        root.removeHandler(handler)
        root.setLevel(old_levels[0])
        cluster_logger.setLevel(old_levels[1])


def test_the_status_lists_every_address_the_node_would_use():
    """To debug a cluster: who the node reaches, where, and what it was configured with."""
    import random

    from app.consensus.auth import MessageSigner
    from app.consensus.messages import Command
    from app.consensus.node import PaxosNode
    from app.consensus.replica import reconfigure_payload
    from app.consensus.runtime import ConsensusRuntime
    from app.consensus.state_machine import ClusterStateMachine
    from app.consensus.storage import MemoryStore

    store = MemoryStore()
    genesis = {"a": "http://a:1", "b": "http://b:1"}
    node = PaxosNode(
        "a", store, ClusterStateMachine(), lambda to, msg: None, random.Random(1), genesis=genesis
    )
    runtime = ConsensusRuntime(
        node, MessageSigner(b"s"), {"seed1": "http://configured:1"}, lambda *a: None
    )
    node.replica.commit(0, 0, Command("c", reconfigure_payload(0, {"a": "http://a:1"})))

    status = runtime.status()
    assert status["addresses"]["a"] == "http://a:1"
    assert status["addresses"]["b"] == "http://b:1"
    assert status["addresses"]["seed1"] == "http://configured:1"
    assert status["seeds"] == {"seed1": "http://configured:1"}
