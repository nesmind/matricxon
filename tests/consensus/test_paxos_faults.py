import random

import pytest

from tests.consensus.sim import SimCluster


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def test_a_new_leader_takes_over_when_the_leader_crashes():
    cluster = SimCluster(seed=10)
    cluster.run(2.0)
    old = cluster.leaders()[0].id
    cluster.submit(old, _set("before", 1))
    cluster.run(1.0)
    cluster.crash(old)
    cluster.run(3.0)
    (new,) = [n.id for n in cluster.leaders()]
    assert new != old
    command = cluster.submit(new, _set("after", 2))
    cluster.run(1.0)
    assert cluster.applied_everywhere(command)

    cluster.restart(old)
    cluster.run(3.0)
    assert cluster.converged() and cluster.machines[old].config == {"before": 1, "after": 2}


def test_a_minority_cannot_decide_anything_and_catches_up_after_the_partition_heals():
    cluster = SimCluster(seed=11)
    cluster.run(2.0)
    cluster.partition({"n0"}, {"n1", "n2"})
    stuck = cluster.submit("n0", _set("minority", True))
    cluster.run(4.0)
    assert not cluster.nodes["n0"].replica.applied(stuck.cmd_id)
    assert len(cluster.leaders()) >= 1 and all(
        n.id != "n0" for n in cluster.leaders() if n.id != "n0"
    )
    majority = cluster.submit("n1", _set("majority", True))
    cluster.run(2.0)
    assert cluster.nodes["n1"].replica.applied(majority.cmd_id) and cluster.nodes[
        "n2"
    ].replica.applied(majority.cmd_id)
    assert not cluster.nodes["n0"].replica.applied(majority.cmd_id)

    cluster.heal()
    cluster.run(5.0)
    assert cluster.converged()
    assert cluster.applied_everywhere(stuck) and cluster.applied_everywhere(majority)


def test_an_isolated_old_leader_cannot_commit_while_the_rest_elect_a_new_one():
    cluster = SimCluster(seed=12)
    cluster.run(2.0)
    old = cluster.leaders()[0].id
    others = {i for i in cluster.ids if i != old}
    cluster.partition({old}, others)
    ghost = cluster.submit(old, _set("ghost", 1))
    cluster.run(3.0)
    assert not cluster.nodes[old].replica.applied(ghost.cmd_id)
    new = next(n for n in cluster.leaders() if n.id in others)
    real = cluster.submit(new.id, _set("real", 2))
    cluster.run(2.0)
    cluster.heal()
    cluster.run(5.0)
    assert cluster.converged() and len(cluster.leaders()) == 1
    assert cluster.applied_everywhere(real) and cluster.applied_everywhere(ghost)
    assert cluster.machines[old].applied.count(_set("ghost", 1)) == 1


def test_commands_survive_a_network_that_drops_duplicates_and_reorders():
    cluster = SimCluster(seed=13, drop=0.25, duplicate=0.15, delay=(0.001, 0.08))
    cluster.run(4.0)
    commands = [cluster.submit(f"n{i % 3}", _set("k", i)) for i in range(25)]
    cluster.run(40.0)
    assert all(cluster.applied_everywhere(c) for c in commands)
    assert cluster.converged() and len(cluster.machines["n0"].applied) == 25


def test_every_node_restarts_from_disk_with_the_same_history():
    cluster = SimCluster(seed=14)
    cluster.run(2.0)
    for i in range(10):
        cluster.submit(f"n{i % 3}", _set(f"k{i}", i))
    cluster.run(3.0)
    before = dict(cluster.machines["n0"].config)
    assert len(before) == 10
    for node_id in cluster.ids:
        cluster.crash(node_id)
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(3.0)
    assert all(m.config == before for m in cluster.machines.values())
    after = cluster.submit("n2", _set("more", 1))
    cluster.run(3.0)
    assert cluster.applied_everywhere(after) and cluster.converged()


def test_a_promise_survives_a_crash():
    """The acceptor must not forget a promise: after a restart it still refuses the lower ballot."""
    from app.consensus.ballot import Ballot
    from app.consensus.messages import Accept, Command, Nack, Prepare, Promise

    cluster = SimCluster(seed=15)
    node = cluster.nodes["n0"]
    assert isinstance(node.acceptor.prepare(Prepare("x", 0, Ballot(9, "x"), 0)), Promise)
    cluster.crash("n0")
    cluster.restart("n0")
    revived = cluster.nodes["n0"]
    refused = revived.acceptor.accept(Accept("y", 0, Ballot(3, "y"), 0, Command("c", 1)))
    assert isinstance(refused, Nack) and refused.promised == Ballot(9, "x")


def test_two_of_three_down_stops_progress_and_one_returning_restores_it():
    cluster = SimCluster(seed=16)
    cluster.run(2.0)
    cluster.crash("n1")
    cluster.crash("n2")
    stuck = cluster.submit("n0", _set("a", 1))
    cluster.run(4.0)
    assert not cluster.nodes["n0"].replica.applied(stuck.cmd_id)
    cluster.restart("n1")
    cluster.run(4.0)
    assert cluster.nodes["n0"].replica.applied(stuck.cmd_id) and cluster.nodes[
        "n1"
    ].replica.applied(stuck.cmd_id)


def test_five_nodes_tolerate_two_failures():
    cluster = SimCluster(seed=17, size=5)
    cluster.run(2.0)
    cluster.crash("n3")
    cluster.crash("n4")
    commands = [cluster.submit(f"n{i % 3}", _set("k", i)) for i in range(10)]
    cluster.run(5.0)
    assert all(cluster.applied_everywhere(c) for c in commands)


@pytest.mark.parametrize("seed", range(60))
def test_random_faults_never_break_safety_and_the_cluster_recovers(seed: int):
    """Random crashes, restarts, partitions and lossy network with commands arriving throughout.
    Safety is checked
    every few events inside run; at the end the faults stop and every command must be applied
    exactly once on every node.
    """
    rng = random.Random(seed)
    size = rng.choice([3, 3, 5])
    cluster = SimCluster(
        seed,
        size=size,
        drop=rng.uniform(0, 0.3),
        duplicate=rng.uniform(0, 0.2),
        delay=(0.001, 0.06),
    )
    cluster.run(1.0)
    for step in range(40):
        action = rng.random()
        if action < 0.2:
            cluster.crash(rng.choice(cluster.ids))
        elif action < 0.4:
            cluster.restart(rng.choice(cluster.ids))
        elif action < 0.55:
            ids = list(cluster.ids)
            rng.shuffle(ids)
            cut = rng.randint(1, size - 1)
            cluster.partition(set(ids[:cut]), set(ids[cut:]))
        elif action < 0.7:
            cluster.heal()
        live = [i for i in cluster.ids if cluster.nodes[i] is not None]
        if live:
            cluster.submit(rng.choice(live), _set(f"k{step % 5}", step))
        cluster.run(rng.uniform(0.05, 1.0))

    cluster.heal()
    cluster.drop = cluster.duplicate = 0.0
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(60.0)

    assert len(cluster.leaders()) == 1
    assert cluster.converged(), f"seed {seed}: nodes did not converge"
    kept = 0
    for command in cluster.submitted:
        if command.cmd_id in cluster.at_risk:
            assert cluster.applied_everywhere(command) or cluster.applied_nowhere(command)
        else:
            assert cluster.applied_everywhere(command), f"seed {seed}: {command} was lost"
        kept += cluster.applied_everywhere(command)
    assert len(cluster.machines[cluster.ids[0]].applied) == kept
