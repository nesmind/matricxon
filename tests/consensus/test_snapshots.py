"""Snapshots: the log stays short; a node asking for log that is gone gets the state instead."""

import random

import pytest

from app.consensus.ballot import Ballot
from app.consensus.config import PaxosConfig
from app.consensus.messages import Command
from app.consensus.storage import DurableState, FileStore
from tests.consensus.sim import SimCluster

EVERY = 20


class SnapCluster(SimCluster):
    """A node that jumped to a snapshot never applied older commands: compares chosen logs only."""

    def __init__(self, seed: int) -> None:
        super().__init__(seed, config=PaxosConfig(snapshot_every=EVERY))

    def check_safety(self) -> None:
        chosen: dict[tuple[int, int], Command] = {}
        for node_id in self.ids:
            for epoch, state in self.stores[node_id].load().epochs.items():
                for slot, command in state.chosen.items():
                    assert chosen.setdefault((epoch, slot), command) == command


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def _fill(cluster: SnapCluster, count: int, start: int = 0) -> None:
    for i in range(start, start + count):
        alive = [n for n in ("n0", "n1", "n2") if cluster.nodes[n] is not None]
        cluster.submit(alive[i % len(alive)], _set(f"k{i}", i))
        cluster.run(0.2)


def _kept(node) -> int:
    return sum(len(slots) for slots in node.replica.chosen.values())


def test_the_log_a_node_keeps_stays_a_few_intervals_long_however_much_is_applied():
    cluster = SnapCluster(seed=70)
    cluster.run(2.0)
    _fill(cluster, 150)
    cluster.run(3.0)
    for node in cluster.live():
        assert node.replica.base is not None
        assert _kept(node) <= 3 * EVERY, _kept(node)
        assert len(node.replica.machine.config) == 150
    assert cluster.converged()


def test_a_node_that_asks_for_log_that_is_gone_gets_the_state_instead():
    cluster = SnapCluster(seed=71)
    cluster.run(2.0)
    _fill(cluster, 120)
    cluster.add_node("n3")
    cluster.run(6.0)
    joiner = cluster.nodes["n3"]
    assert len(joiner.replica.machine.config) == 120
    assert len(cluster.machines["n3"].applied) < 120
    late = cluster.submit("n0", _set("after", True))
    cluster.run(3.0)
    assert joiner.replica.applied(late.cmd_id) and cluster.converged()


def test_a_node_restarted_from_its_compacted_journal_picks_up_where_it_was():
    cluster = SnapCluster(seed=72)
    cluster.run(2.0)
    _fill(cluster, 90)
    cluster.crash("n1")
    _fill(cluster, 30, start=90)
    cluster.restart("n1")
    cluster.run(6.0)
    assert len(cluster.nodes["n1"].replica.machine.config) == 120 and cluster.converged()


def test_a_compacted_journal_file_reloads_to_its_snapshot_and_the_log_after_it(tmp_path):
    state = DurableState(genesis={"a": "x"}, snapshot={"epoch": 1, "slot": 5})
    state.epoch(1).promised = Ballot(3, "a")
    state.epoch(1).chosen = {5: Command("c5", 1), 6: Command("c6", 2)}
    state.epoch(1).accepted = {6: (Ballot(3, "a"), Command("c6", 2))}
    store = FileStore(tmp_path / "j")
    store.compact(state)
    store.save_chosen(1, 7, Command("c7", 3))
    loaded = FileStore(tmp_path / "j").load()
    assert loaded.snapshot == {"epoch": 1, "slot": 5} and loaded.genesis == {"a": "x"}
    assert sorted(loaded.epoch(1).chosen) == [5, 6, 7] and loaded.epoch(1).promised == Ballot(
        3, "a"
    )


class FaultyCluster(SnapCluster):
    def __init__(self, seed: int, size: int, rng: random.Random) -> None:
        SimCluster.__init__(
            self,
            seed,
            size=size,
            drop=rng.uniform(0, 0.4),
            duplicate=rng.uniform(0, 0.3),
            delay=(0.001, 0.1),
            config=PaxosConfig(snapshot_every=5),
        )


@pytest.mark.parametrize("seed", range(8))
def test_faults_with_constant_snapshots_neither_diverge_nor_lose_a_command(seed: int):
    rng = random.Random(seed)
    size = rng.choice([3, 5])
    cluster = FaultyCluster(seed, size, rng)
    cluster.run(1.0)
    for step in range(120):
        action = rng.random()
        if action < 0.2:
            cluster.crash(rng.choice(cluster.ids))
        elif action < 0.45:
            cluster.restart(rng.choice(cluster.ids))
        elif action < 0.6:
            ids = list(cluster.ids)
            rng.shuffle(ids)
            cut = rng.randint(1, size - 1)
            cluster.partition(set(ids[:cut]), set(ids[cut:]))
        elif action < 0.75:
            cluster.heal()
        live = [i for i in cluster.ids if cluster.nodes[i] is not None]
        if live:
            cluster.submit(rng.choice(live), _set(f"k{step % 9}", step))
        cluster.run(rng.uniform(0.02, 1.0))
    cluster.heal()
    cluster.drop = cluster.duplicate = 0.0
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(90.0)
    assert len(cluster.leaders()) == 1 and cluster.converged()
    for command in cluster.submitted:
        assert cluster.applied_everywhere(command) or (
            command.cmd_id in cluster.at_risk and cluster.applied_nowhere(command)
        )
