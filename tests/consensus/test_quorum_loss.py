"""A cluster that loses its majority: the leader steps down, status says so, campaigns back off."""

import random

from app.consensus.config import PaxosConfig
from app.consensus.quorum import QuorumWatch
from tests.consensus.sim import SimCluster


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def _status(cluster: SimCluster, node_id: str) -> tuple[bool, int | None]:
    return cluster.nodes[node_id].quorum_status(cluster.now)


def test_a_healthy_idle_cluster_never_steps_down_and_reports_a_quorum_everywhere():
    cluster = SimCluster(seed=50)
    cluster.run(3.0)
    leader = cluster.leaders()[0].id
    for _ in range(40):
        cluster.run(0.5)
        assert [n.id for n in cluster.leaders()] == [leader]
    assert _status(cluster, leader) == (True, 3)
    assert all(_status(cluster, n)[0] for n in cluster.ids if n != leader)


def test_a_leader_cut_off_from_everyone_steps_down_and_stops_claiming_a_quorum():
    cluster = SimCluster(seed=51)
    cluster.run(3.0)
    old = cluster.leaders()[0].id
    others = {i for i in cluster.ids if i != old}
    cluster.partition({old}, others)
    cluster.run(0.1)
    assert cluster.nodes[old].is_leader
    cluster.run(PaxosConfig().election_timeout * 3)
    assert not cluster.nodes[old].is_leader
    assert _status(cluster, old)[0] is False
    (new,) = [n.id for n in cluster.leaders()]
    assert new in others and _status(cluster, new)[0]


def test_a_leader_with_a_dead_majority_steps_down_and_reports_what_it_can_reach():
    cluster = SimCluster(seed=52)
    cluster.run(3.0)
    leader = cluster.leaders()[0].id
    for node_id in cluster.ids:
        if node_id != leader:
            cluster.crash(node_id)
    cluster.run(PaxosConfig().election_timeout * 4)
    node = cluster.nodes[leader]
    assert not node.is_leader and _status(cluster, leader)[0] is False
    stuck = cluster.submit(leader, _set("during", 1))
    cluster.run(5.0)
    assert not node.replica.applied(stuck.cmd_id)


def test_a_leader_that_keeps_a_majority_does_not_step_down_when_one_follower_is_lost():
    cluster = SimCluster(seed=53)
    cluster.run(3.0)
    leader = cluster.leaders()[0].id
    gone = next(i for i in cluster.ids if i != leader)
    cluster.crash(gone)
    cluster.run(10.0)
    assert cluster.nodes[leader].is_leader
    assert _status(cluster, leader) == (True, 2)


def test_recovery_after_a_long_outage_still_happens_within_the_backoff_limit():
    config = PaxosConfig()
    cluster = SimCluster(seed=54)
    cluster.run(3.0)
    cluster.partition({"n0"}, {"n1"}, {"n2"})
    cluster.run(60.0)
    assert cluster.leaders() == []
    cluster.heal()
    cluster.run(config.election_timeout * (2 * QuorumWatch.MAX_BACKOFF + 4))
    assert len(cluster.leaders()) == 1
    command = cluster.submit(cluster.leaders()[0].id, _set("after", 1))
    cluster.run(2.0)
    assert cluster.applied_everywhere(command)


def test_a_lone_follower_campaigns_less_and_less_often():
    quiet = SimCluster(seed=55)
    quiet.run(3.0)
    follower = next(i for i in quiet.ids if i != quiet.leaders()[0].id)
    for node_id in quiet.ids:
        if node_id != follower:
            quiet.crash(node_id)
    node = quiet.nodes[follower]
    start = node._max_round
    quiet.run(10.0)
    # without backoff this reached round 24 in 10 s; the waits double up to 4x
    assert node._max_round - start <= 10


def test_backoff_doubles_to_its_cap_and_resets_when_a_leader_is_found():
    watch = QuorumWatch(1.0, random.Random(0))
    waits = [watch.next_campaign_wait() for _ in range(6)]
    assert 1.0 <= waits[0] < 2.0 and 2.0 <= waits[1] < 4.0 and 4.0 <= waits[2] < 8.0
    assert all(4.0 <= w < 8.0 for w in waits[2:])
    watch.leader_found()
    assert 1.0 <= watch.next_campaign_wait() < 2.0


def test_the_watch_counts_only_recent_answers_and_gives_a_new_leader_time_to_settle():
    watch = QuorumWatch(1.0, random.Random(0))
    watch.became_leader(10.0)
    voters = ["a", "b", "c"]
    assert not watch.must_step_down("a", voters, 2, 10.5)
    assert watch.must_step_down("a", voters, 2, 11.2)
    watch.heard_from("b", 11.1)
    assert watch.reachable("a", voters, 11.2) == 2 and not watch.must_step_down(
        "a", voters, 2, 11.2
    )
    assert watch.reachable("a", voters, 12.3) == 1
