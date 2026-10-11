"""The cluster doctor: each way a cluster can break is named, with the fix when a tool can do it."""

from pathlib import Path

from app.cluster.doctor import ClusterDoctor, Probe, same_node
from app.cluster.doctor_cli import load, main

A, B, C = "http://a:8420", "http://b:8420", "http://c:8420"


def up(url: str, node_id: str, voters: list[str], **extra) -> Probe:
    members = {v: f"http://{v}:8420" for v in voters}
    status = {
        "enabled": True,
        "node_id": node_id,
        "epoch": 3,
        "role": "follower",
        "has_quorum": True,
        "voters": voters,
        "members": members,
        "nodes": members,
        "removed": [],
        "applied_total": 1000,
        "cluster_id": "c" * 32,
        "target_voters": 3,
    } | extra
    return Probe(url, status, {"node_id": node_id, "checks": []})


def down(url: str) -> Probe:
    return Probe(url, None, None)


def codes(*probes: Probe, previous: dict | None = None) -> dict[str, str]:
    return {f.code: f.action for f in ClusterDoctor(list(probes), previous).findings()}


def test_a_healthy_cluster_has_no_findings():
    voters = ["a", "b", "c"]
    probes = [up(A, "a", voters, role="leader"), up(B, "b", voters), up(C, "c", voters)]
    assert ClusterDoctor(probes).findings() == []


def test_no_answer_at_all_and_no_consensus_are_told_apart():
    assert list(codes(down(A), down(B))) == ["ALL_DOWN"]
    off = Probe(A, {"enabled": False}, {"node_id": "x", "checks": []})
    assert list(codes(off)) == ["NO_CONSENSUS"]


def test_a_down_voter_is_dropped_through_the_leader_while_a_majority_is_left():
    voters = ["a", "b", "c"]
    found = codes(up(A, "a", voters, role="leader"), up(B, "b", voters), down(C))
    assert found["DOWN"] == "" and found["VOTER_DOWN"] == "drop-voter http://a:8420 c"


def test_a_voter_nobody_probed_is_a_ghost():
    voters = ["a", "b", "ghost"]
    found = codes(up(A, "a", voters, role="leader"), up(B, "b", voters))
    assert found["GHOST_VOTER"] == "drop-voter http://a:8420 ghost"


def test_no_majority_forces_a_group_of_one_from_the_most_up_to_date_survivor():
    voters = ["a", "b", "c", "d", "e"]
    behind = up(A, "a", voters, epoch=3, applied_total=900, has_quorum=False)
    ahead = up(B, "b", voters, epoch=3, applied_total=990, has_quorum=False)
    gone = [down(C), down("http://d:8420"), down("http://e:8420")]
    assert codes(behind, ahead, *gone)["NO_QUORUM"] == "recover http://b:8420 b"
    assert "UNPROBED_VOTERS" in codes(behind, ahead, down(C))
    lone = codes(
        Probe(
            A,
            {
                "enabled": True,
                "node_id": "n",
                "epoch": 1,
                "role": "off",
                "voters": ["x", "y"],
                "applied_total": 5,
            },
            None,
        ),
        down(B),
    )
    assert "ALL_VOTERS_LOST" in lone and lone["ALL_VOTERS_LOST"] == ""


def test_a_voter_whose_address_moved_is_updated():
    voters = ["a", "b", "c"]
    moved = up(B, "b", voters, advertise_url="http://new:8420")
    found = codes(up(A, "a", voters, role="leader"), moved, up(C, "c", voters))
    assert found["ADDRESS_MOVED"] == "add-voter http://a:8420 b http://new:8420"


def test_a_learner_is_replaying_then_stuck_when_it_does_not_move_and_is_reset():
    voters = ["a"]
    learner = up(B, "b", voters, applied_total=100)
    cluster = [up(A, "a", voters, role="leader", target_voters=1), learner]
    assert codes(*cluster)["REPLAYING"] == ""
    assert codes(*cluster, previous={B: 100})["STUCK_LEARNER"] == "reset-node http://b:8420"


def test_a_caught_up_learner_is_made_a_voter_when_the_cluster_has_too_few():
    registered = {"a": "http://a:8420", "b": "http://b:8420"}
    found = codes(up(A, "a", ["a"], role="leader", nodes=registered), up(B, "b", ["a"]))
    assert found["UNDER_VOTERS"] == "add-voter http://a:8420 b http://b:8420"


def test_a_node_removed_on_purpose_that_still_runs_is_reset():
    voters = ["a"]
    found = codes(
        up(A, "a", voters, role="leader", removed=["b"]), up(B, "b", voters, removed=["b"])
    )
    assert found["REMOVED_BUT_RUNNING"] == "reset-node http://b:8420"


def test_two_clusters_failing_storage_and_node_notes_are_reported():
    voters = ["a", "b"]
    other = up(B, "b", voters, cluster_id="d" * 32, note="the journal disk is slow")
    other.ready["checks"] = [{"id": "models", "status": "action_needed"}]
    found = codes(up(A, "a", voters, role="leader"), other)
    assert {"TWO_CLUSTERS", "SHARED_STORAGE", "NOTE"} <= set(found)


def test_localhost_names_are_one_machine():
    assert same_node("http://localhost:8420/", "http://127.0.0.1:8420")
    assert not same_node("http://a:8420", "http://a:8421")


def test_the_command_line_prints_findings_and_fails_when_something_is_wrong(tmp_path: Path, capsys):
    (tmp_path / "0.url").write_text(A)
    (tmp_path / "1.url").write_text(B)
    (tmp_path / "0.status").write_text('{"enabled": false}')
    assert [p.url for p in load(tmp_path)] == [A, B]
    assert main([str(tmp_path)]) == 1
    assert "DOWN" in capsys.readouterr().out


def _paths(probe: Probe) -> Probe:
    probe.status["config"] = {"cluster/paths": {"models_dir": "/m", "prompt_cache_dir": "/c"}}
    return probe


def test_a_rearrangement_keeps_the_best_voter_drops_the_other_voters_and_resets_the_rest():
    from app.cluster.rearrange import RearrangeRefused, plan

    voters = ["a", "b", "c"]
    old = _paths(up(A, "a", voters, role="leader", epoch=3, applied_total=900))
    best = _paths(up(B, "b", voters, epoch=4, applied_total=950))
    fresh = Probe(C, {"enabled": False}, {"node_id": "z", "checks": []})
    result = plan([old, best, fresh, down("http://d:8420")])
    assert result.seed.url == B and result.quorum_via == A and result.drop == ["a", "c"]
    assert [p.url for p in result.others] == [A, C] and result.down == ["http://d:8420"]
    assert result.advertise(fresh) == C and result.models == "/m" and result.cache == "/c"
    assert result.lines()[0] == f"seed\t{B}\tb\thttp://b:8420"
    try:
        plan([up(A, "a", voters)])
    except RearrangeRefused as refusal:
        assert "shared folders" in str(refusal)
    try:
        plan([down(A)])
    except RearrangeRefused as refusal:
        assert "no node runs consensus" in str(refusal)


def test_the_other_nodes_are_found_from_the_one_that_answers_but_not_the_one_asked_again():
    from app.cluster.doctor_cli import discover

    voters = ["a", "b", "c"]
    me = up(
        "http://localhost:8420",
        "a",
        voters,
        nodes={"a": "http://lab:8420/", "b": B, "c": C},
        members={"a": "http://lab:8420", "b": B},
    )
    assert discover([me]) == [B, C]
    assert discover([me, up(B, "b", voters)]) == [C]
    assert discover([down(A)]) == []
