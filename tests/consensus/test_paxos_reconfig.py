"""Changing the voters while the cluster runs: epochs, joiners that catch up, and the faults
around it."""

import random

import pytest

from app.consensus.config import PaxosConfig
from tests.consensus.sim import SimCluster


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def _settle(cluster: SimCluster, seconds: float = 6.0) -> None:
    cluster.run(seconds)


def test_a_new_node_catches_up_then_becomes_a_voter():
    cluster = SimCluster(seed=30)
    cluster.run(2.0)
    early = [cluster.submit(f"n{i % 3}", _set(f"k{i}", i)) for i in range(8)]
    cluster.run(2.0)

    cluster.add_node("n3")
    cluster.run(4.0)
    joiner = cluster.nodes["n3"]
    assert not joiner.is_voter and not joiner.is_leader
    assert all(joiner.replica.applied(c.cmd_id) for c in early)

    done = cluster.reconfigure(["n0", "n1", "n2", "n3"])
    _settle(cluster)
    assert cluster.nodes["n3"].is_voter and cluster.nodes["n3"].epoch == 1
    assert cluster.voters() == ["n0", "n1", "n2", "n3"]
    assert cluster.nodes["n0"].replica.result(done.cmd_id) is True

    late = cluster.submit("n3", _set("from_the_new_voter", True))
    _settle(cluster)
    assert cluster.applied_everywhere(late) and cluster.converged()


def test_the_quorum_follows_the_voters():
    cluster = SimCluster(seed=31)
    cluster.run(2.0)
    cluster.add_node("n3")
    cluster.run(3.0)
    cluster.reconfigure(["n0", "n1", "n2", "n3"])
    _settle(cluster)
    assert cluster.nodes["n0"].quorum == 3
    cluster.crash("n2")
    cluster.crash("n3")
    stuck = cluster.submit("n0", _set("needs_three", 1))
    _settle(cluster)
    assert not cluster.nodes["n0"].replica.applied(stuck.cmd_id)
    cluster.restart("n3")
    _settle(cluster)
    assert cluster.nodes["n0"].replica.applied(stuck.cmd_id)


def test_removing_a_follower_shrinks_the_quorum():
    cluster = SimCluster(seed=32, size=5)
    cluster.run(2.0)
    cluster.reconfigure(["n0", "n1", "n2"])
    _settle(cluster)
    assert cluster.voters() == ["n0", "n1", "n2"]
    cluster.crash("n3")
    cluster.crash("n4")
    cluster.crash("n2")
    command = cluster.submit("n0", _set("two_of_three", 1))
    _settle(cluster)
    assert cluster.nodes["n0"].replica.applied(command.cmd_id)


def test_a_leader_can_remove_itself():
    cluster = SimCluster(seed=33)
    cluster.run(2.0)
    old = cluster.leaders()[0].id
    rest = [i for i in cluster.ids if i != old]
    cluster.reconfigure(rest)
    _settle(cluster)
    assert not cluster.nodes[old].is_voter and not cluster.nodes[old].is_leader
    (new,) = cluster.leaders()
    assert new.id in rest
    command = cluster.submit(new.id, _set("after_removal", 1))
    _settle(cluster)
    assert cluster.applied_everywhere(command)
    assert cluster.nodes[old].replica.applied(command.cmd_id)


def test_a_dead_node_can_be_replaced():
    cluster = SimCluster(seed=34)
    cluster.run(2.0)
    cluster.crash("n2")
    cluster.add_node("n3")
    cluster.run(3.0)
    cluster.reconfigure(["n0", "n1", "n3"])
    _settle(cluster)
    assert cluster.voters() == ["n0", "n1", "n3"]
    cluster.crash("n0")
    command = cluster.submit("n1", _set("after_replacement", 1))
    _settle(cluster)
    assert cluster.nodes["n1"].replica.applied(command.cmd_id) and cluster.nodes[
        "n3"
    ].replica.applied(command.cmd_id)


def test_every_voter_can_be_replaced():
    cluster = SimCluster(seed=35)
    cluster.run(2.0)
    before = [cluster.submit("n0", _set(f"old{i}", i)) for i in range(5)]
    cluster.run(2.0)
    for new in ("n3", "n4", "n5"):
        cluster.add_node(new)
    cluster.run(5.0)
    cluster.reconfigure(["n3", "n4", "n5"])
    _settle(cluster, 8.0)
    assert cluster.voters() == ["n3", "n4", "n5"]
    for old in ("n0", "n1", "n2"):
        cluster.crash(old)
    after = cluster.submit("n4", _set("new_era", 1))
    _settle(cluster)
    assert all(cluster.nodes[n].replica.applied(after.cmd_id) for n in ("n3", "n4", "n5"))
    assert all(cluster.nodes["n3"].replica.applied(c.cmd_id) for c in before)


def test_two_competing_reconfigurations_only_one_wins():
    cluster = SimCluster(seed=36, size=5)
    cluster.run(2.0)
    first = cluster.nodes["n0"].reconfigure(
        {i: f"sim://{i}" for i in ("n0", "n1", "n2")}, cluster.now
    )
    second = cluster.nodes["n3"].reconfigure(
        {i: f"sim://{i}" for i in ("n2", "n3", "n4")}, cluster.now
    )
    cluster.submitted += [first, second]
    _settle(cluster, 8.0)
    node = cluster.nodes["n0"]
    decided = [
        node.replica.result(c.cmd_id) for c in (first, second) if node.replica.applied(c.cmd_id)
    ]
    assert decided.count(True) == 1
    assert cluster.voters() in (["n0", "n1", "n2"], ["n2", "n3", "n4"])
    assert cluster.converged()


def test_commands_sent_during_a_reconfiguration_are_applied_exactly_once():
    cluster = SimCluster(seed=37)
    cluster.run(2.0)
    cluster.add_node("n3")
    cluster.run(3.0)
    commands = []
    for i in range(20):
        if i == 7:
            cluster.reconfigure(["n0", "n1", "n2", "n3"])
        commands.append(cluster.submit(f"n{i % 3}", _set("k", i)))
        cluster.run(0.05)
    _settle(cluster, 10.0)
    assert all(cluster.applied_everywhere(c) for c in commands)
    assert len(cluster.machines["n0"].applied) == 20


def test_a_node_cut_off_during_the_change_catches_up_across_epochs():
    cluster = SimCluster(seed=38)
    cluster.run(2.0)
    cluster.partition({"n2"}, {"n0", "n1"})
    cluster.add_node("n3")
    cluster.run(3.0)
    cluster.reconfigure(["n0", "n1", "n3"])
    _settle(cluster)
    cluster.submit("n0", _set("while_away", 1))
    _settle(cluster)
    assert cluster.nodes["n2"].epoch == 0
    cluster.heal()
    _settle(cluster, 10.0)
    assert cluster.nodes["n2"].epoch == 1 and not cluster.nodes["n2"].is_voter
    assert cluster.machines["n2"].config == cluster.machines["n0"].config


def test_a_node_that_is_never_made_a_voter_stays_a_quiet_learner():
    cluster = SimCluster(seed=39)
    cluster.run(2.0)
    cluster.add_node("n3")
    commands = [cluster.submit("n0", _set(f"k{i}", i)) for i in range(5)]
    cluster.run(15.0)
    learner = cluster.nodes["n3"]
    assert learner.role == "follower" and not learner.is_voter and learner.epoch == 0
    assert all(learner.replica.applied(c.cmd_id) for c in commands)
    assert len(cluster.leaders()) == 1


def test_the_history_and_the_voters_survive_a_full_restart():
    cluster = SimCluster(seed=40)
    cluster.run(2.0)
    cluster.add_node("n3")
    cluster.run(3.0)
    cluster.reconfigure(["n0", "n1", "n3"])
    _settle(cluster)
    cluster.submit("n0", _set("kept", 1))
    _settle(cluster)
    for node_id in cluster.ids:
        cluster.crash(node_id)
    for node_id in cluster.ids:
        cluster.restart(node_id)
    _settle(cluster)
    assert cluster.voters() == ["n0", "n1", "n3"] and cluster.nodes["n3"].is_voter
    assert all(m.config == {"kept": 1} for m in cluster.machines.values())
    after = cluster.submit("n3", _set("after", 1))
    _settle(cluster)
    assert cluster.applied_everywhere(after)


@pytest.mark.parametrize("seed", range(40))
def test_random_faults_and_reconfigurations_keep_the_cluster_safe_and_live(seed: int):
    """Random crashes, partitions, loss, new nodes, voters added and removed while commands keep
    arriving. Safety is
    checked throughout; at the end the faults stop and what was acknowledged to a surviving voter
    must be applied exactly once everywhere.
    """
    rng = random.Random(1000 + seed)
    cluster = SimCluster(
        seed, size=3, drop=rng.uniform(0, 0.25), duplicate=rng.uniform(0, 0.15), delay=(0.001, 0.06)
    )
    spare = ["n3", "n4", "n5", "n6"]
    cluster.run(1.0)
    for step in range(45):
        action = rng.random()
        live = [i for i in cluster.ids if cluster.nodes[i] is not None]
        if action < 0.12:
            cluster.crash(rng.choice(cluster.ids))
        elif action < 0.25:
            cluster.restart(rng.choice(cluster.ids))
        elif action < 0.35:
            ids = list(cluster.ids)
            rng.shuffle(ids)
            cut = rng.randint(1, len(ids) - 1)
            cluster.partition(set(ids[:cut]), set(ids[cut:]))
        elif action < 0.45:
            cluster.heal()
        elif action < 0.55 and spare:
            cluster.add_node(spare.pop(0))
        elif action < 0.75 and live:
            pool = [i for i in cluster.ids]
            members = rng.sample(pool, rng.randint(2, min(5, len(pool))))
            voter = [n for n in cluster.live() if n.is_voter]
            if voter:
                cluster.submitted.append(
                    voter[0].reconfigure({m: f"sim://{m}" for m in members}, cluster.now)
                )
        voters = [n.id for n in cluster.live() if n.is_voter]
        if voters:
            cluster.submit(rng.choice(voters), _set(f"k{step % 5}", step))
        cluster.run(rng.uniform(0.05, 0.8))

    cluster.heal()
    cluster.drop = cluster.duplicate = 0.0
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(60.0)

    for node in cluster.live():
        if not node.is_voter:
            cluster.at_risk.update(c.cmd_id for c in node._pending.open())
    assert len(cluster.leaders()) == 1, f"seed {seed}: {len(cluster.leaders())} leaders"
    assert cluster.converged(), f"seed {seed}: nodes did not converge"
    for command in cluster.submitted:
        if command.cmd_id in cluster.at_risk:
            assert cluster.applied_everywhere(command) or cluster.applied_nowhere(command)
        else:
            assert cluster.applied_everywhere(command), f"seed {seed}: {command} lost"


def test_a_reconfiguration_that_lost_the_race_is_retried_into_the_next_epoch_and_does_nothing():
    cluster = SimCluster(seed=41, size=5)
    cluster.run(2.0)
    first = cluster.nodes["n0"].reconfigure(
        {i: f"sim://{i}" for i in ("n0", "n1", "n2", "n3")}, cluster.now
    )
    second = cluster.nodes["n1"].reconfigure(
        {i: f"sim://{i}" for i in ("n0", "n1", "n2", "n4")}, cluster.now
    )
    cluster.submitted += [first, second]
    _settle(cluster, 10.0)
    node = cluster.nodes["n0"]
    # n1 is a voter in both sets, so it kept retrying the loser until applied, as a no-op
    assert {node.replica.result(first.cmd_id), node.replica.result(second.cmd_id)} == {True, False}
    assert node.epoch == 1
    assert cluster.converged()


def test_a_node_in_a_newer_epoch_ignores_a_stale_leader_and_tells_it_to_catch_up():
    from app.consensus.ballot import Ballot
    from app.consensus.messages import Accept, Command, Heartbeat

    cluster = SimCluster(seed=42)
    cluster.run(2.0)
    cluster.reconfigure(["n0", "n1", "n2"])
    _settle(cluster)
    node = cluster.nodes["n1"]
    assert node.epoch == 1
    sent = []
    node._send_out = lambda to, msg: sent.append((to, msg))
    promised = node.acceptor.promised
    node.receive(
        Accept("n2", 0, Ballot(99, "n2"), 7, Command("old", {"op": "set", "key": "x", "value": 1})),
        cluster.now,
    )
    assert node.acceptor.promised == promised and not node.replica.chosen.get(1)
    assert [type(m).__name__ for _, m in sent] == ["Heartbeat"]
    assert isinstance(sent[0][1], Heartbeat) and sent[0][1].epoch == 1 and sent[0][0] == "n2"


def test_a_new_node_replays_many_epochs_at_the_real_timers_without_waiting_between_them():
    """With the real-network timers (0.5 s retry) a node waiting for its timer between epochs needed
    ~0.5 s per epoch
    (more when asking a dead address from old history). Finishing an epoch now asks the same peer
    for the next at once.
    """
    config = PaxosConfig(heartbeat=0.2, election_timeout=1.0, retry_interval=0.5)
    cluster = SimCluster(seed=70, size=3, config=config)
    cluster.run(4.0)
    for i in range(8):
        cluster.reconfigure(["n0", "n1"] if i % 2 == 0 else ["n0", "n1", "n2"])
        cluster.run(1.5)
    target = max(n.epoch for n in cluster.live())
    assert target >= 8
    cluster.add_node("n3")
    start = cluster.now
    while cluster.nodes["n3"].epoch < target and cluster.now - start < 30:
        cluster.run(0.1)
    assert cluster.nodes["n3"].epoch == target
    assert cluster.now - start < 2.5


def test_a_voter_far_behind_across_epochs_catches_up_fast_with_the_real_timers():
    """Bug: two voters, one far behind across epochs. The lagging node still
    believed it was a voter
    and never pulled the log, and the other (no quorum, so no leader, so no heartbeats) reached it
    only once per
    election timeout: one batch of 100 commands per 5-10 seconds, minutes without a quorum.
    """
    config = PaxosConfig(heartbeat=1.0, election_timeout=5.0, retry_interval=2.5)
    cluster = SimCluster(seed=41, config=config)
    cluster.run(8.0)
    cluster.reconfigure(["n0"])
    cluster.run(8.0)
    cluster.partition({"n1"}, {"n0", "n2"})
    for i in range(900):
        cluster.submit("n0", _set(f"a{i}", i))
        if i % 100 == 99:
            cluster.run(1.0)
    cluster.reconfigure(["n0", "n1"])
    cluster.run(8.0)
    assert cluster.nodes["n1"].epoch < cluster.nodes["n0"].epoch

    cluster.heal()
    cluster.run(40.0)

    assert cluster.nodes["n1"].epoch == cluster.nodes["n0"].epoch
    assert cluster.machines["n1"].config == cluster.machines["n0"].config
    assert len(cluster.leaders()) == 1


def test_a_demoted_voter_that_nobody_tells_finds_out_by_asking():
    """Bug: a voter was demoted but never heard. A leader talks only to current
    voters and the one
    message announcing the change was lost, so the node stayed a voter of an old epoch, campaigned
    for an hour and
    never caught up. Here nobody may START a conversation with it (no heartbeats, proposals or
    campaigns): it must ask, and the answers get through.
    """
    config = PaxosConfig(heartbeat=1.0, election_timeout=5.0, retry_interval=2.5)
    cluster = SimCluster(seed=42, config=config)
    cluster.run(8.0)
    cluster.partition({"n1"}, {"n0", "n2"})
    cluster.reconfigure(["n0", "n2"])
    cluster.run(8.0)
    cluster.heal()
    real_send = cluster._send

    def only_answers(src: str, dst: str, msg) -> None:
        if dst == "n1" and type(msg).__name__ in {"Heartbeat", "Accept", "Prepare"}:
            return
        real_send(src, dst, msg)

    cluster._send = only_answers  # type: ignore[method-assign]
    assert cluster.nodes["n1"].epoch == 0 and cluster.nodes["n1"].is_voter

    cluster.run(40.0)

    assert cluster.nodes["n1"].epoch == cluster.nodes["n0"].epoch == 1
    assert not cluster.nodes["n1"].is_voter
    assert cluster.machines["n1"].config == cluster.machines["n0"].config
