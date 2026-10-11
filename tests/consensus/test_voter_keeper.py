"""VoterKeeper: the leader keeps the cluster at the target number of voters by itself."""

import threading
from types import SimpleNamespace

from app.consensus.learners import Learners
from app.consensus.voter_keeper import VoterKeeper


class FakeRuntime:
    """Just what the keeper touches: a leader with some voters, learners it has heard from."""

    def __init__(
        self, voters: list[str], silent: tuple[str, ...] = (), leader: bool = True
    ) -> None:
        self.now = 1000.0
        self._lock = threading.RLock()
        self.learners = Learners()
        self.shared_config = dict
        self.submitted: list[dict] = []
        self.calls: list[tuple] = []
        watch = SimpleNamespace(heard_within=lambda v, now: v not in self.silent)
        self.silent = silent
        self.node = SimpleNamespace(
            id=voters[0],
            role="leader" if leader else "follower",
            members=voters,
            replica=SimpleNamespace(epoch=2, next_slot=500),
            quorum_status=lambda now: (True, len(voters)),
            _watch=watch,
        )

    def submit(self, payload: dict) -> None:
        self.submitted.append(payload)

    def _clock(self) -> float:
        return self.now

    def learner(self, node_id: str, behind: int = 0, epoch: int = 2) -> None:
        self.learners.seen(node_id, f"http://{node_id}:8420", epoch, 500 - behind, self.now)

    def add_voter(self, node_id: str, address: str) -> bool:
        self.calls.append(("add", node_id, address))
        self.node.members = [*self.node.members, node_id]
        return True

    def remove_voter(self, node_id: str) -> bool:
        self.calls.append(("remove", node_id))
        self.node.members = [v for v in self.node.members if v != node_id]
        return True


def test_a_small_cluster_promotes_every_learner_that_has_caught_up():
    runtime = FakeRuntime(["n1"])
    runtime.learner("n2", behind=3)
    done = VoterKeeper(runtime, target=3).step()
    assert runtime.calls == [("add", "n2", "http://n2:8420")] and len(done) == 1


def test_it_stops_at_the_target_and_later_nodes_stay_learners():
    runtime = FakeRuntime(["n1", "n2"])
    for name in ("n3", "n4", "n5"):
        runtime.learner(name)
    VoterKeeper(runtime, target=3).step()
    assert [c[1] for c in runtime.calls] == ["n3"]
    assert VoterKeeper(runtime, target=3).step() == []


def test_the_closest_learner_goes_first_and_one_far_behind_waits():
    runtime = FakeRuntime(["n1"])
    runtime.learner("far", behind=400)
    runtime.learner("near", behind=1)
    runtime.learner("mid", behind=20)
    VoterKeeper(runtime, target=3).step()
    assert [c[1] for c in runtime.calls] == ["near", "mid"]


def test_a_learner_of_another_epoch_or_gone_silent_is_not_promoted():
    runtime = FakeRuntime(["n1"])
    runtime.learner("old-epoch", epoch=1)
    runtime.learners.seen("silent", "http://silent:8420", 2, 500, runtime.now - 60)
    assert VoterKeeper(runtime, target=3).step() == [] and not runtime.calls


def test_only_the_leader_acts_and_target_zero_changes_no_voter():
    follower = FakeRuntime(["n1"], leader=False)
    follower.learner("n2")
    assert VoterKeeper(follower, target=3).step() == []
    runtime = FakeRuntime(["n1"])
    runtime.learner("n2")
    assert VoterKeeper(runtime, target=0).step() == [] and not runtime.calls


def test_a_voter_silent_for_good_is_replaced_by_a_caught_up_learner():
    runtime = FakeRuntime(["n1", "n2", "n3"], silent=("n3",))
    runtime.learner("n4")
    keeper = VoterKeeper(runtime, target=3, lost_after=120.0)
    assert keeper.step() == []
    runtime.now += 119
    runtime.learner("n4")
    assert keeper.step() == []
    runtime.now += 2
    runtime.learner("n4")
    assert len(keeper.step()) == 1
    assert runtime.calls == [("add", "n4", "http://n4:8420"), ("remove", "n3")]


def test_a_voter_that_answers_again_is_not_replaced():
    runtime = FakeRuntime(["n1", "n2", "n3"], silent=("n3",))
    runtime.learner("n4")
    keeper = VoterKeeper(runtime, target=3, lost_after=120.0)
    keeper.step()
    runtime.now += 100
    runtime.silent = ()
    runtime.learner("n4")
    keeper.step()
    runtime.now += 100
    runtime.silent = ("n3",)
    runtime.learner("n4")
    assert keeper.step() == [] and not runtime.calls


def test_a_node_removed_on_purpose_is_not_promoted_again_while_it_still_follows_the_log():
    runtime = FakeRuntime(["n1"])
    runtime.learner("n2")
    runtime.shared_config = lambda: {"cluster/removed": ["n2"]}
    assert VoterKeeper(runtime, target=3).step() == [] and not runtime.calls
    runtime.shared_config = lambda: {"cluster/removed": []}
    assert len(VoterKeeper(runtime, target=3).step()) == 1


def test_the_leader_registers_every_learner_it_hears_from_in_the_clusters_own_node_list():
    runtime = FakeRuntime(["n1", "n2", "n3"])
    runtime.learner("n4", behind=300)
    runtime.learner("n5")
    runtime.shared_config = lambda: {"cluster/removed": ["n5"], "cluster/nodes": {}}
    VoterKeeper(runtime, target=3).step()
    assert runtime.submitted == [
        {
            "op": "set",
            "key": "cluster/nodes",
            "value": {"n4": "http://n4:8420"},
        }
    ]
    assert not runtime.calls
    runtime.submitted.clear()
    runtime.shared_config = lambda: {"cluster/nodes": {"n4": "http://n4:8420"}}
    VoterKeeper(runtime, target=3).step()
    assert runtime.submitted == [
        {
            "op": "set",
            "key": "cluster/nodes",
            "value": {"n4": "http://n4:8420", "n5": "http://n5:8420"},
        }
    ]


def test_with_target_zero_the_node_list_is_still_kept_but_no_voter_is_changed():
    runtime = FakeRuntime(["n1"])
    runtime.learner("n2")
    VoterKeeper(runtime, target=0).step()
    assert runtime.submitted and not runtime.calls


def test_a_registered_node_whose_address_now_answers_as_someone_else_leaves_the_node_list():
    """A node that was reset got a new identity: its old id must not stay in the cluster's list."""
    runtime = FakeRuntime(["n1"])
    runtime.shared_config = lambda: {
        "cluster/nodes": {"old": "http://reset:8420", "down": "http://down:8420"}
    }
    keeper = VoterKeeper(runtime, target=3, stranger=lambda node_id, address: node_id == "old")
    keeper.step()
    assert runtime.submitted == [
        {"op": "set", "key": "cluster/nodes", "value": {"down": "http://down:8420"}}
    ]
    runtime.submitted.clear()
    keeper.step()
    assert runtime.submitted == []


def test_a_node_heard_from_lately_is_never_probed():
    runtime = FakeRuntime(["n1"])
    runtime.learner("n2")
    runtime.shared_config = lambda: {"cluster/nodes": {"n2": "http://n2:8420"}}
    VoterKeeper(runtime, target=0, stranger=lambda *_: pytest_fail()).step()


def pytest_fail():
    raise AssertionError("a node that is being heard from must not be probed")


def test_a_voter_taken_out_on_purpose_is_marked_before_the_change_so_its_exit_cannot_lose_it():
    from app.consensus.voters import VoterChanges

    order: list[str] = []
    runtime = SimpleNamespace(
        shared_config=lambda: {"cluster/removed": ["old"], "cluster/nodes": {"n2": "http://n2"}},
        submit=lambda payload: order.append(payload["key"]),
    )
    changes = VoterChanges(runtime)
    changes._change = lambda *args, **kwargs: order.append("change") or True
    assert changes.remove("n2", 5.0, leave=True)
    assert order == ["cluster/removed", "cluster/nodes", "change"]
    order.clear()
    changes._change = lambda *args, **kwargs: order.append("change") or False
    assert not changes.remove("n2", 5.0, leave=True)
    assert order == ["cluster/removed", "cluster/nodes", "change", "cluster/removed"]


def test_the_leader_makes_the_cluster_id_once_and_a_follower_never():
    import threading

    from app.consensus.cluster_id import KEY, ClusterIdKeeper

    class Runtime:
        def __init__(self, role: str, config: dict) -> None:
            self._lock = threading.RLock()
            self.node = SimpleNamespace(role=role, quorum_status=lambda now: (True, 1))
            self.config, self.sent = config, []

        def _clock(self) -> float:
            return 0.0

        def shared_config(self) -> dict:
            return self.config

        def submit(self, payload: dict) -> tuple:
            self.sent.append(payload)
            self.config[KEY] = payload["value"]
            return None, True, True

    follower = Runtime("follower", {})
    assert not ClusterIdKeeper(follower).step() and follower.sent == []
    leader = Runtime("leader", {})
    assert ClusterIdKeeper(leader).step()
    assert leader.sent[0]["op"] == "cas" and leader.sent[0]["expected"] is None
    assert len(leader.config[KEY]) == 32
    assert ClusterIdKeeper(leader).step() and len(leader.sent) == 1
