from pathlib import Path

import pytest

from app.consensus.acceptor import Acceptor
from app.consensus.ballot import ZERO, Ballot
from app.consensus.messages import NOOP, Accept, Accepted, Command, Nack, Prepare, Promise
from app.consensus.proposer import Campaign, Leadership
from app.consensus.replica import Replica, SafetyViolation
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import FileStore, MemoryStore

B1, B2, B3 = Ballot(1, "a"), Ballot(2, "a"), Ballot(2, "b")


def cmd(name: str, payload: object = None) -> Command:
    return Command(name, payload if payload is not None else {"op": "set", "key": name, "value": 1})


def _acceptor() -> tuple[Acceptor, MemoryStore]:
    store = MemoryStore()
    return Acceptor("x", 0, store.load().epoch(0), store), store


def test_ballots_order_by_round_then_node():
    assert ZERO < B1 < B2 < B3
    assert Ballot(2, "b") > Ballot(2, "a") and Ballot(3, "a") > Ballot(2, "z")


def test_an_acceptor_promises_only_upward_and_saves_each_promise():
    acceptor, store = _acceptor()
    assert isinstance(acceptor.prepare(Prepare("a", 0, B2, 0)), Promise)
    refused = acceptor.prepare(Prepare("a", 0, B1, 0))
    assert isinstance(refused, Nack) and refused.promised == B2
    assert store.load().epoch(0).promised == B2
    assert isinstance(acceptor.prepare(Prepare("a", 0, B2, 0)), Promise)


def test_an_acceptor_refuses_an_accept_below_its_promise_and_takes_one_at_or_above():
    acceptor, store = _acceptor()
    acceptor.prepare(Prepare("a", 0, B2, 0))
    assert isinstance(acceptor.accept(Accept("a", 0, B1, 0, cmd("old"))), Nack)
    assert store.load().epoch(0).accepted == {}
    assert isinstance(acceptor.accept(Accept("a", 0, B2, 0, cmd("now"))), Accepted)
    assert isinstance(acceptor.accept(Accept("b", 0, B3, 1, cmd("higher"))), Accepted)
    assert acceptor.promised == B3
    assert store.load().epoch(0).accepted == {0: (B2, cmd("now")), 1: (B3, cmd("higher"))}


def test_a_promise_reports_what_was_accepted_from_the_asked_slot_on():
    acceptor, _ = _acceptor()
    for slot in range(4):
        acceptor.accept(Accept("a", 0, B1, slot, cmd(f"c{slot}")))
    promise = acceptor.prepare(Prepare("b", 0, B2, 2))
    assert [slot for slot, _, _ in promise.accepted] == [2, 3]


def test_the_campaign_completes_once_and_keeps_the_value_with_the_highest_ballot():
    campaign = Campaign(B3, quorum=2)
    assert not campaign.add("a", ((0, B1, cmd("old")), (2, B1, cmd("gap-end"))))
    assert campaign.add("b", ((0, B2, cmd("newer")),))
    assert not campaign.add("c", ())
    plan = campaign.plan(from_slot=0, chosen={})
    assert plan == {
        0: cmd("newer"),
        1: NOOP,
        2: cmd("gap-end"),
    }


def test_the_plan_skips_slots_already_known_to_be_chosen():
    campaign = Campaign(B3, quorum=1)
    campaign.add("a", ((0, B1, cmd("x")), (1, B1, cmd("y"))))
    assert campaign.plan(0, chosen={0: cmd("x")}) == {1: cmd("y")}
    empty = Campaign(B3, quorum=1)
    empty.add("a", ())
    assert empty.plan(0, {}) == {}


def test_a_leader_counts_each_slot_to_a_majority_once():
    leadership = Leadership(B3, quorum=2, first_free_slot=5)
    assert leadership.next_slot_for(cmd("a")) == 5 and leadership.next_slot_for(cmd("b")) == 6
    assert leadership.accepted(5, "x") is None
    assert leadership.accepted(5, "x") is None
    assert leadership.accepted(5, "y") == cmd("a")
    assert leadership.accepted(5, "z") is None
    assert [slot for slot, _, _ in leadership.unchosen()] == [6]


def _replica(store: MemoryStore | None = None) -> Replica:
    store = store or MemoryStore()
    return Replica(ClusterStateMachine(), store, store.load())


def test_the_replica_applies_in_order_once_the_gap_is_filled():
    replica = _replica()
    replica.commit(0, 1, cmd("second", {"op": "set", "key": "k", "value": "second"}))
    assert replica.next_slot == 0 and replica.machine.config == {}
    replica.commit(0, 0, cmd("first", {"op": "set", "key": "k", "value": "first"}))
    assert replica.next_slot == 2 and replica.machine.config == {"k": "second"}


def test_the_replica_skips_a_command_id_it_already_applied_and_noops():
    replica = _replica()
    increment = cmd("same", {"op": "cas", "key": "n", "expected": None, "value": 1})
    replica.commit(0, 0, increment)
    replica.commit(0, 1, NOOP)
    replica.commit(0, 2, increment)
    assert replica.next_slot == 3 and replica.machine.config == {"n": 1}
    assert replica.result("same") is True and replica.applied("same")


def test_the_replica_raises_if_two_different_commands_are_chosen_for_a_slot():
    replica = _replica()
    replica.commit(0, 0, cmd("a"))
    assert replica.commit(0, 0, cmd("a")) is False
    with pytest.raises(SafetyViolation):
        replica.commit(0, 0, cmd("b"))


def test_a_replica_rebuilds_its_state_from_the_store():
    store = MemoryStore()
    first = _replica(store)
    first.commit(0, 0, cmd("a", {"op": "set", "key": "x", "value": 1}))
    first.commit(0, 1, cmd("b", {"op": "add_node", "id": "n9", "address": "http://n9"}))
    again = _replica(store)
    assert (
        again.next_slot == 2
        and again.machine.config == {"x": 1}
        and again.machine.nodes == {"n9": "http://n9"}
    )


def test_file_store_round_trips_and_ignores_a_torn_last_line(tmp_path: Path):
    path = tmp_path / "paxos" / "journal"
    store = FileStore(path)
    store.save_promise(0, B2)
    store.save_accept(0, 3, B2, cmd("a", {"op": "set", "key": "k", "value": [1, 2]}))
    store.save_chosen(0, 3, cmd("a", {"op": "set", "key": "k", "value": [1, 2]}))
    with path.open("a") as handle:
        handle.write('{"k": "p", "b": [9')
    state = FileStore(path).load()
    assert state.epoch(0).promised == B2
    assert state.epoch(0).accepted[3][0] == B2 and state.epoch(0).accepted[3][1].cmd_id == "a"
    assert state.epoch(0).chosen[3].payload == {"op": "set", "key": "k", "value": [1, 2]}
    assert FileStore(tmp_path / "none").load().epochs == {}


def test_the_state_machine_operations():
    machine = ClusterStateMachine()
    assert machine.apply({"op": "set", "key": "a", "value": 1}) is True
    assert machine.apply({"op": "cas", "key": "a", "expected": 2, "value": 3}) is False
    assert machine.apply({"op": "cas", "key": "a", "expected": 1, "value": 3}) is True
    assert (
        machine.apply({"op": "delete", "key": "a"}) is True
        and machine.apply({"op": "delete", "key": "a"}) is False
    )
    assert machine.apply({"op": "add_node", "id": "n1", "address": "x"}) and machine.nodes == {
        "n1": "x"
    }
    assert machine.apply({"op": "remove_node", "id": "n1"}) and machine.nodes == {}
    with pytest.raises(ValueError):
        machine.apply({"op": "nope"})


def test_a_sealed_epoch_serves_its_history_only_up_to_the_command_that_sealed_it():
    from app.consensus.replica import reconfigure_payload

    replica = _replica()
    replica.commit(0, 0, cmd("a"))
    replica.commit(0, 1, cmd("seal", reconfigure_payload(0, {"n0": "x", "n1": "y"})))
    assert replica.epoch == 1 and replica.members == {"n0": "x", "n1": "y"}
    replica.commit(0, 2, cmd("void"))
    replica.commit(1, 0, cmd("fresh"))
    assert [slot for slot, _ in replica.chosen_from(0, 0)] == [0, 1]
    assert [slot for slot, _ in replica.chosen_from(1, 0)] == [0]
    assert "void" not in [c.cmd_id for c in replica.chosen[1].values()]
    assert replica.machine.config == {"a": 1, "fresh": 1} and not replica.applied("void")


def test_a_reconfigure_aimed_at_an_older_epoch_does_nothing():
    from app.consensus.replica import reconfigure_payload

    replica = _replica()
    replica.commit(0, 0, cmd("first", reconfigure_payload(0, {"n0": "x"})))
    assert replica.epoch == 1
    replica.commit(1, 0, cmd("late", reconfigure_payload(0, {"n9": "z"})))
    assert replica.epoch == 1 and replica.members == {"n0": "x"} and replica.next_slot == 1
    assert replica.result("late") is False and replica.result("first") is True


def test_a_resent_accept_is_not_journaled_again(tmp_path):
    from app.consensus.acceptor import Acceptor
    from app.consensus.ballot import Ballot
    from app.consensus.messages import Accept, Command
    from app.consensus.storage import FileStore

    path = tmp_path / "j"
    store = FileStore(path)
    acceptor = Acceptor("n0", 0, store.load().epoch(0), store)
    accept = Accept("n1", 0, Ballot(1, "n1"), 3, Command("c", {"op": "set"}))
    for _ in range(50):
        assert type(acceptor.accept(accept)).__name__ == "Accepted"
    assert len(path.read_bytes().rstrip(b"\0").splitlines()) == 2


def test_a_bloated_journal_is_compacted_without_changing_the_state(tmp_path):
    import json

    from app.consensus.storage import FileStore

    path = tmp_path / "j"
    lines = [{"k": "g", "m": {"n0": "u"}}, {"k": "p", "e": 0, "b": [1, "n0"]}]
    lines += [{"k": "a", "e": 0, "s": 7, "b": [1, "n0"], "c": ["c7", {"v": 1}]}] * 5000
    lines += [{"k": "c", "e": 0, "s": 7, "c": ["c7", {"v": 1}]}]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    before = FileStore(path).load()
    assert len(path.read_bytes().rstrip(b"\0").splitlines()) == 4
    after = FileStore(path).load()
    assert after.genesis == before.genesis and after.epochs == before.epochs
    assert after.epochs[0].chosen[7].cmd_id == "c7" and after.epochs[0].promised.round == 1


def test_a_torn_last_line_is_cut_off_and_later_writes_survive_a_restart(tmp_path):
    from app.consensus.messages import Command
    from app.consensus.storage import FileStore

    path = tmp_path / "j"
    store = FileStore(path)
    store.save_chosen(0, 0, Command("a", 1))
    store.save_chosen(0, 1, Command("b", 2))
    raw = path.read_bytes().rstrip(b"\0")
    path.write_bytes(raw[:-9] + b"\0" * 50)

    again = FileStore(path)
    assert set(again.load().epochs[0].chosen) == {0}
    again.save_chosen(0, 1, Command("b", 2))
    assert set(FileStore(path).load().epochs[0].chosen) == {0, 1}


def test_the_padding_grows_in_chunks_and_never_changes_what_is_read(tmp_path):
    from app.consensus.messages import Command
    from app.consensus.storage import FileStore

    path = tmp_path / "j"
    store = FileStore(path)
    store.CHUNK = 256
    for slot in range(40):
        store.save_chosen(0, slot, Command(f"c{slot}", {"v": slot}))
    assert path.stat().st_size % 256 == 0 and path.stat().st_size > 40 * 40
    assert len(FileStore(path).load().epochs[0].chosen) == 40


def test_a_node_that_left_the_voters_keeps_its_address():
    """The leader must still reach a demoted or removed node (to tell it that it is behind)."""
    from app.consensus.messages import Command
    from app.consensus.replica import Replica, reconfigure_payload
    from app.consensus.state_machine import ClusterStateMachine
    from app.consensus.storage import MemoryStore

    store = MemoryStore()
    store.save_genesis({"a": "http://a:1", "b": "http://b:1"})
    replica = Replica(ClusterStateMachine(), store, store.load())
    replica.commit(0, 0, Command("c1", reconfigure_payload(0, {"a": "http://a:1"})))

    assert "b" not in replica.members
    assert replica.address_of("b", {}) == "http://b:1"
    assert replica.address_of("b", {"b": "http://configured:1"}) == "http://configured:1"
    assert replica.address_of("a", {"a": "http://elsewhere:1"}) == "http://a:1"
    assert replica.address_of("nobody", {}) is None


class _FakeReplica:
    """Just where a node stands in the log."""

    def __init__(self) -> None:
        self.epoch, self.next_slot = 0, 0


def test_a_node_behind_asks_again_only_when_the_batch_is_taken_in_or_stalls():
    import random

    from app.consensus.catchup import CatchUpPolicy

    here = _FakeReplica()
    policy = CatchUpPolicy("n", here, random.Random(1), 0.1, batch=100)
    policy.note_ahead("leader", 0, 1000)
    first = policy.pull(0.0)
    assert first is not None and first[1].from_slot == 0
    here.next_slot = 7
    assert policy.pull(0.2) is None
    here.next_slot = 60
    assert policy.pull(0.4) is None
    here.next_slot = 100
    second = policy.pull(0.5)
    assert second is not None and second[1].from_slot == 100
    assert policy.pull(1.0) is None
    again = policy.pull(1.6)
    assert again is not None and again[1].from_slot == 100


def test_applied_total_counts_the_slots_of_every_epoch_so_far():
    replica = _replica()
    replica.commit(0, 0, cmd("a"))
    replica.commit(0, 1, cmd("b"))
    assert replica.applied_total() == 2
    replica.commit(
        0, 2, Command("r", {"op": "reconfigure", "epoch": 0, "members": {"x": "http://x"}})
    )
    assert replica.epoch == 1 and replica.applied_total() == 2
