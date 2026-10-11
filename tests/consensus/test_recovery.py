"""Forcing a new group of one when the other voters are gone for good."""

import os
from pathlib import Path

import pytest

from app.consensus.ballot import Ballot
from app.consensus.messages import Command
from app.consensus.recovery import GroupRecovery, RecoveryRefusedError, run
from app.consensus.storage import FileStore, MemoryStore
from tests.consensus.sim import SimCluster


def _set(key: str, value: object) -> dict:
    return {"op": "set", "key": key, "value": value}


def _lost_two_of_three() -> tuple[SimCluster, list]:
    cluster = SimCluster(seed=77)
    cluster.run(2.0)
    done = [cluster.submit("n0", _set(f"k{i}", i)) for i in range(6)]
    cluster.run(3.0)
    cluster.crash("n1")
    cluster.crash("n2")
    cluster.crash("n0")
    return cluster, done


def test_a_stuck_cluster_is_revived_from_the_survivor_without_losing_what_it_knew():
    cluster, done = _lost_two_of_three()
    plan = GroupRecovery(cluster.stores["n0"]).force_new_group("n0")
    assert plan.voters.keys() == {"n0", "n1", "n2"}

    cluster.restart("n0")
    cluster.run(3.0)
    node = cluster.nodes["n0"]
    assert node.members == ["n0"] and node.epoch == 1 and node.is_leader
    assert all(node.replica.applied(c.cmd_id) for c in done)

    late = cluster.submit("n0", _set("after", True))
    cluster.run(1.0)
    assert node.replica.applied(late.cmd_id)


def test_fresh_nodes_join_the_recovered_group():
    cluster, _ = _lost_two_of_three()
    GroupRecovery(cluster.stores["n0"]).force_new_group("n0")
    cluster.restart("n0")
    cluster.run(2.0)
    for name in ("n3", "n4"):
        cluster.add_node(name)
    cluster.run(3.0)
    cluster.reconfigure(["n0", "n3", "n4"])
    cluster.run(6.0)
    assert cluster.nodes["n3"].is_voter and cluster.nodes["n4"].is_voter
    after = cluster.submit("n3", _set("three_again", 3))
    cluster.run(3.0)
    assert all(cluster.nodes[n].replica.applied(after.cmd_id) for n in ("n0", "n3", "n4"))


def test_the_plan_counts_entries_accepted_but_not_known_chosen():
    store = MemoryStore()
    store.save_genesis({"n0": "u0", "n1": "u1", "n2": "u2"})
    store.save_accept(0, 0, Ballot(1, "n1"), Command("a", _set("x", 1)))
    store.save_accept(0, 1, Ballot(1, "n1"), Command("b", _set("y", 2)))
    store.save_chosen(0, 0, Command("a", _set("x", 1)))
    plan = GroupRecovery(store).plan("n0")
    assert plan.problem is None and plan.next_slot == 1 and plan.unconfirmed == 1


@pytest.mark.parametrize("who", ["stranger", "n0"])
def test_it_refuses_what_makes_no_sense(who: str):
    cluster = SimCluster(seed=9, size=1)
    cluster.run(1.0)
    with pytest.raises(RecoveryRefusedError):
        GroupRecovery(cluster.stores["n0"]).force_new_group(who)


def test_it_refuses_an_empty_journal():
    assert GroupRecovery(MemoryStore()).plan("n0").problem is not None


def test_it_works_on_a_file_journal(tmp_path: Path):
    cluster, _ = _lost_two_of_three()
    path = tmp_path / "consensus.journal"
    store = FileStore(path)
    state = cluster.stores["n0"].load()
    store.save_genesis(state.genesis)
    for epoch, ep in state.epochs.items():
        for slot, command in ep.chosen.items():
            store.save_chosen(epoch, slot, command)
    GroupRecovery(FileStore(path)).force_new_group("n0")
    assert GroupRecovery(FileStore(path)).plan("n0").problem == (
        "this node is already a group of one: nothing to recover"
    )


def _journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    from app.config import settings

    monkeypatch.setattr(settings, "node_state_dir", tmp_path)
    monkeypatch.setattr(settings, "consensus_node_id", "n0")
    path = tmp_path / "consensus.journal"
    path.parent.mkdir(parents=True, exist_ok=True)
    FileStore(path).save_genesis({"n0": "u0", "n1": "u1", "n2": "u2"})
    return path


def test_the_command_defaults_to_a_dry_run_and_needs_the_id(tmp_path, monkeypatch):
    path = _journal(tmp_path, monkeypatch)
    before, lines = path.read_bytes(), []
    assert run([], lines.append) == 0
    assert run(["--force-new-group"], lines.append) == 1
    assert run(["--force-new-group", "--confirm", "n1"], lines.append) == 1
    assert path.read_bytes() == before and not list(tmp_path.rglob("*.bak-*"))


def test_the_command_backs_up_then_forces_the_group(tmp_path, monkeypatch):
    path = _journal(tmp_path, monkeypatch)
    assert run(["--force-new-group", "--confirm", "n0"], lambda _: None) == 0
    assert len(list(path.parent.glob("consensus.journal.bak-*"))) == 1
    assert GroupRecovery(FileStore(path)).plan("n0").problem is not None


def test_the_command_refuses_while_the_server_runs(tmp_path, monkeypatch):
    path = _journal(tmp_path, monkeypatch)
    (tmp_path / "pid").write_text(str(os.getpid()))
    before = path.read_bytes()
    assert run(["--force-new-group", "--confirm", "n0"], lambda _: None) == 1
    assert path.read_bytes() == before
