from tests.consensus.sim import SimCluster


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def test_one_leader_emerges_and_a_command_reaches_every_node():
    cluster = SimCluster(seed=1)
    cluster.run(2.0)
    assert len(cluster.leaders()) == 1
    command = cluster.submit("n1", _set("keep_alive", 600))
    cluster.run(1.0)
    assert cluster.applied_everywhere(command)
    assert {m.config["keep_alive"] for m in cluster.machines.values()} == {600}


def test_commands_are_applied_in_the_same_order_everywhere():
    cluster = SimCluster(seed=2)
    cluster.run(2.0)
    for i in range(30):
        cluster.submit(f"n{i % 3}", _set("counter", i))
    cluster.run(3.0)
    assert cluster.converged()
    assert len(cluster.machines["n0"].applied) == 30
    assert (
        cluster.machines["n0"].applied
        == cluster.machines["n1"].applied
        == cluster.machines["n2"].applied
    )


def test_a_command_submitted_twice_is_applied_once():
    cluster = SimCluster(seed=3)
    cluster.run(2.0)
    command = cluster.nodes["n0"].new_command(_set("a", 1))
    for node_id in ("n0", "n1", "n2", "n1"):
        cluster.nodes[node_id].submit(command, cluster.now)
    cluster.run(2.0)
    assert [m.applied for m in cluster.machines.values()] == [[_set("a", 1)]] * 3


def test_a_command_submitted_before_any_leader_exists_is_not_lost():
    cluster = SimCluster(seed=4)
    command = cluster.submit("n2", _set("early", True))
    cluster.run(3.0)
    assert cluster.applied_everywhere(command)


def test_compare_and_set_decides_between_two_competing_nodes():
    cluster = SimCluster(seed=5)
    cluster.run(2.0)
    first = cluster.submit("n0", {"op": "cas", "key": "lease", "expected": None, "value": "n0"})
    second = cluster.submit("n1", {"op": "cas", "key": "lease", "expected": None, "value": "n1"})
    cluster.run(2.0)
    results = sorted(
        [
            cluster.nodes["n0"].replica.result(first.cmd_id),
            cluster.nodes["n0"].replica.result(second.cmd_id),
        ]
    )
    assert results == [False, True]
    assert len({m.config["lease"] for m in cluster.machines.values()}) == 1
