"""The wire format, the signatures, and a real-threaded 3-node cluster talking through them."""

import json
import random
import threading
import time

import pytest

from app.consensus import messages as m
from app.consensus.auth import MessageSigner
from app.consensus.ballot import Ballot
from app.consensus.codec import CodecError, decode, encode
from app.consensus.config import PaxosConfig
from app.consensus.node import PaxosNode
from app.consensus.runtime import SIGNATURE_HEADER, BadSignatureError, ConsensusRuntime
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import MemoryStore

B = Ballot(3, "n1")
C = m.Command("c1", {"op": "set", "key": "k", "value": [1, {"a": None}]})
SAMPLES = [
    m.Prepare("n1", 2, B, 5),
    m.Promise("n1", 2, B, ((5, Ballot(1, "n0"), C), (6, Ballot(2, "n2"), m.NOOP))),
    m.Promise("n1", 0, B, ()),
    m.Accept("n1", 2, B, 7, C),
    m.Accepted("n1", 2, B, 7),
    m.Nack("n1", 2, B, Ballot(9, "n2")),
    m.Commit("n1", 2, 7, C),
    m.Heartbeat("n1", 2, B, 12),
    m.HeartbeatAck("n2", 2, B),
    m.CatchUp("n1", 1, 0),
    m.Snapshot("n1", 2, 7, {"epoch": 2, "slot": 7, "machine": {"config": {"k": 1}, "nodes": {}}}),
    m.Forward("n1", C),
]


@pytest.mark.parametrize("msg", SAMPLES, ids=lambda msg: type(msg).__name__)
def test_every_message_survives_the_wire(msg):
    wire = json.loads(json.dumps(encode(msg)))
    assert decode(wire) == msg


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"type": "Nope"},
        {"type": "Accept", "sender": "a"},
        {"type": "Prepare", "sender": "a", "epoch": 0, "ballot": "x", "from_slot": 0},
        {"type": "Commit", "sender": "a", "epoch": 0, "slot": 1, "command": 5},
    ],
)
def test_a_malformed_message_is_an_error_not_a_guess(bad):
    with pytest.raises(CodecError):
        decode(bad)


def test_signatures_bind_the_exact_bytes_and_the_secret():
    signer = MessageSigner(b"shared")
    signature = signer.sign(b"hello")
    assert signer.verify(b"hello", signature)
    assert (
        not signer.verify(b"hellp", signature)
        and not signer.verify(b"hello", "")
        and not signer.verify(b"hello", "00")
    )
    assert not MessageSigner(b"other").verify(b"hello", signature)
    with pytest.raises(ValueError):
        MessageSigner(b"")


class Hub:
    """Stands in for the network: delivers a posted body to the runtime owning the url, unless that
    node is 'down'."""

    def __init__(self) -> None:
        self.runtimes: dict[str, ConsensusRuntime] = {}
        self.down: set[str] = set()
        self.hangs: set[str] = set()
        self.rejected = 0

    def post(self, url: str, body: bytes, headers: dict[str, str]) -> None:
        node_id = url.split("//")[1].split("/")[0]
        runtime = self.runtimes.get(node_id)
        if node_id in self.hangs:
            time.sleep(1.2)
            raise TimeoutError(url)
        if runtime is None or node_id in self.down:
            raise ConnectionError(url)
        try:
            runtime.handle(body, headers[SIGNATURE_HEADER])
        except (BadSignatureError, CodecError):
            self.rejected += 1


def _runtime(
    hub: Hub, node_id: str, genesis: dict[str, str], secret: bytes = b"shared"
) -> ConsensusRuntime:
    node = PaxosNode(
        node_id,
        MemoryStore(),
        ClusterStateMachine(),
        lambda to, msg: None,
        random.Random(node_id),
        PaxosConfig(heartbeat=0.03, election_timeout=0.15, retry_interval=0.1),
        genesis=genesis,
    )
    runtime = ConsensusRuntime(node, MessageSigner(secret), genesis, hub.post)
    hub.runtimes[node_id] = runtime
    return runtime


def _wait(condition, timeout: float = 8.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def cluster():
    hub = Hub()
    genesis = {name: f"http://{name}" for name in ("a", "b", "c")}
    runtimes = {name: _runtime(hub, name, genesis) for name in genesis}
    for runtime in runtimes.values():
        runtime.start()
    yield hub, runtimes
    for runtime in runtimes.values():
        runtime.stop()


def _leaders(runtimes, alive=None):
    return [
        n
        for n, r in runtimes.items()
        if (alive is None or n in alive) and r.status()["role"] == "leader"
    ]


def test_three_real_threaded_nodes_elect_a_leader_and_replicate_through_signed_messages(cluster):
    hub, runtimes = cluster
    assert _wait(lambda: len(_leaders(runtimes)) == 1)
    command, applied, result = runtimes["c"].submit(
        {"op": "set", "key": "keep_alive", "value": 600}
    )
    assert applied and result is True
    assert _wait(
        lambda: all(r.status()["config"].get("keep_alive") == 600 for r in runtimes.values())
    )
    assert hub.rejected == 0


def test_the_cluster_keeps_working_when_the_leader_goes_away(cluster):
    hub, runtimes = cluster
    assert _wait(lambda: len(_leaders(runtimes)) == 1)
    old = _leaders(runtimes)[0]
    runtimes[old].submit({"op": "set", "key": "before", "value": 1})
    hub.down.add(old)
    runtimes[old].stop()
    alive = [n for n in runtimes if n != old]
    assert _wait(lambda: len(_leaders(runtimes, alive)) == 1)
    survivor = alive[0]
    _, applied, _ = runtimes[survivor].submit({"op": "set", "key": "after", "value": 2})
    assert applied
    assert all(runtimes[n].status()["config"] == {"before": 1, "after": 2} for n in alive)


def test_a_node_with_the_wrong_secret_is_ignored(cluster):
    hub, runtimes = cluster
    assert _wait(lambda: len(_leaders(runtimes)) == 1)
    intruder = _runtime(
        hub, "x", {"a": "http://a", "b": "http://b", "c": "http://c"}, secret=b"guess"
    )
    hub.runtimes.pop("x")
    body = json.dumps(
        encode(m.Commit("x", 0, 0, m.Command("evil", {"op": "set", "key": "evil", "value": 1})))
    ).encode()
    with pytest.raises(BadSignatureError):
        runtimes["a"].handle(body, intruder._signer.sign(body))
    with pytest.raises(BadSignatureError):
        runtimes["a"].handle(body, "")
    time.sleep(0.3)
    assert all("evil" not in r.status()["config"] for r in runtimes.values())


def test_a_tampered_body_is_rejected(cluster):
    _, runtimes = cluster
    signer = MessageSigner(b"shared")
    body = json.dumps(encode(m.CatchUp("b", 0, 0))).encode()
    signature = signer.sign(body)
    with pytest.raises(BadSignatureError):
        runtimes["a"].handle(body.replace(b'"b"', b'"c"'), signature)


def test_stopping_a_runtime_ends_its_threads():
    hub = Hub()
    runtime = _runtime(hub, "a", {"a": "http://a"})
    runtime.start()
    assert _wait(lambda: runtime.status()["role"] == "leader")
    runtime.stop()
    assert not any(t.is_alive() for t in threading.enumerate() if t.name.startswith("consensus-"))


def test_a_peer_that_hangs_does_not_hold_up_the_others():
    """Regression: with one shared sender thread, messages to a partitioned node (each hanging until
    timeout) queued
    ahead of those to healthy ones, and no election finished in time.
    """
    hub = Hub()
    genesis = {name: f"http://{name}" for name in ("a", "b", "c")}
    runtimes = {name: _runtime(hub, name, genesis) for name in ("a", "b")}
    hub.hangs.add("c")
    for runtime in runtimes.values():
        runtime.start()
    try:
        assert _wait(lambda: len(_leaders(runtimes)) == 1, timeout=6.0)
        _, applied, _ = runtimes["b"].submit({"op": "set", "key": "k", "value": 1}, wait=4.0)
        assert applied
    finally:
        for runtime in runtimes.values():
            runtime.stop()


def test_a_full_queue_drops_the_oldest_message_not_the_newest():
    from app.consensus.runtime import _PeerQueue

    release = threading.Event()
    seen: list[int] = []

    def slow_send(peer: str, msgs: list[object]) -> None:
        release.wait(3.0)
        seen.extend(msgs)

    peer = _PeerQueue("p", slow_send)
    for i in range(_PeerQueue.LIMIT * 2):
        peer.put(i)
    release.set()
    peer.close()
    assert len(seen) <= _PeerQueue.LIMIT + _PeerQueue.BATCH
    assert seen[-1] == _PeerQueue.LIMIT * 2 - 1


def test_every_message_type_is_known_to_the_codec_and_covered_here():
    """A message type missing from the codec is rejected by every receiver (a leader never hearing
    acks steps down for
    good): list them from the module itself so none is forgotten.
    """
    import dataclasses

    declared = {
        cls for cls in vars(m).values()
        if isinstance(cls, type) and dataclasses.is_dataclass(cls) and cls not in (m.Command,)
    }  # fmt: skip
    from app.consensus.codec import _TYPES

    assert declared == set(_TYPES.values())
    assert declared == {type(sample) for sample in SAMPLES}


def test_a_node_whose_address_does_not_answer_as_it_is_not_made_a_voter():
    """Bug: a voter listed as http://localhost:8420 was the other node itself."""
    hub = Hub()
    genesis = {"n0": "sim://n0"}
    runtime = _runtime(hub, "n0", genesis)
    asked: list[tuple[str, str]] = []
    runtime._probe = lambda node_id, address: asked.append((node_id, address)) or False
    runtime.start()
    try:
        assert runtime.add_voter("n1", "http://localhost:8420", wait=0.5) is False
        assert asked == [("n1", "http://localhost:8420")]
        assert runtime.status()["voters"] == ["n0"]
    finally:
        runtime.stop()


def test_the_probe_accepts_only_the_node_that_answers_with_its_own_id():
    import httpx

    from app.consensus.probe import answers_as

    real_get = httpx.get
    try:
        httpx.get = lambda url, timeout: httpx.Response(200, json={"node_id": "n1"})
        assert answers_as("n1", "http://a:8420") is True
        assert answers_as("n2", "http://a:8420") is False
        httpx.get = lambda url, timeout: httpx.Response(503)
        assert answers_as("n1", "http://a:8420") is False

        def refused(url, timeout):
            raise httpx.ConnectError("down")

        httpx.get = refused
        assert answers_as("n1", "http://a:8420") is False
    finally:
        httpx.get = real_get


def test_a_voter_that_moved_gets_its_new_address_recorded_with_the_same_voters():
    """A node that moved (a real address, not a lab one): the voter list is what others use."""
    hub = Hub()
    runtime = _runtime(hub, "n0", {"n0": "http://lab:8420"})
    runtime._probe = lambda node_id, address: address.startswith("http://real")
    runtime.start()
    try:
        assert runtime.status()["members"] == {"n0": "http://lab:8420"}
        epoch = runtime.status()["epoch"]

        assert runtime.add_voter("n0", "http://real:8420", wait=8) is True
        status = runtime.status()
        assert status["members"] == {"n0": "http://real:8420"} and status["voters"] == ["n0"]
        assert status["epoch"] == epoch + 1

        assert runtime.add_voter("n0", "http://real:8420", wait=2) is True
        assert runtime.status()["epoch"] == epoch + 1

        assert runtime.add_voter("n0", "http://elsewhere:8420", wait=2) is False
        assert runtime.status()["members"] == {"n0": "http://real:8420"}
    finally:
        runtime.stop()


def _bare_runtime(secret: bytes = b"s", own: str | None = None) -> ConsensusRuntime:
    node = PaxosNode(
        "leader",
        MemoryStore(),
        ClusterStateMachine(),
        lambda to, msg: None,
        random.Random(1),
        PaxosConfig(),
        genesis={"leader": "http://leader"},
    )
    return ConsensusRuntime(node, MessageSigner(secret), {}, lambda *a: None, own_address=own)


def test_a_signed_sender_address_makes_an_unknown_node_reachable():
    leader, learner = _bare_runtime(), _bare_runtime(own="http://learner:8420/")
    body = json.dumps(encode(m.CatchUp("learner", 0, 0))).encode()
    headers = learner._headers(body)
    assert leader.address_of("learner") is None
    leader.handle(body, *headers.values())
    assert leader.address_of("learner") == "http://learner:8420"


def test_an_address_with_a_bad_signature_is_ignored_and_a_voters_own_record_wins():
    leader, learner = _bare_runtime(), _bare_runtime(own="http://learner:8420")
    body = json.dumps(encode(m.CatchUp("learner", 0, 0))).encode()
    sig, address, _ = learner._headers(body).values()
    leader.handle(body, sig, "http://evil:1", "0" * 64)
    assert leader.address_of("learner") is None
    body = json.dumps(encode(m.CatchUp("leader", 0, 0))).encode()
    sign = leader._signer.sign
    leader.handle(body, sign(body), "http://elsewhere:1", sign(b"http://elsewhere:1"))
    assert leader.address_of("leader") == "http://leader"


def test_the_leader_learns_who_follows_the_log_from_their_catch_up_requests():
    leader, learner = _bare_runtime(), _bare_runtime(own="http://learner:8420")
    body = json.dumps(encode(m.CatchUp("learner", 0, 7))).encode()
    leader.handle(body, *learner._headers(body).values())
    now = leader._clock()
    assert leader.learners.caught_up(["leader"], 0, 10, 50, now) == [
        ("learner", "http://learner:8420")
    ]
    assert leader.learners.caught_up(["leader"], 0, 500, 50, now) == []


def test_a_posted_list_of_messages_is_taken_in_one_by_one():
    signer = MessageSigner(b"shared")
    runtime = _runtime(Hub(), "a", {"a": "http://a"})
    seen: list[object] = []
    runtime.node.receive = lambda msg, now: seen.append(msg)
    msgs = [m.Heartbeat("b", 0, Ballot(0, ""), n) for n in range(3)]
    body = json.dumps([encode(m) for m in msgs]).encode()
    runtime.handle(body, signer.sign(body))
    assert seen == msgs


def test_a_waiting_backlog_goes_out_as_one_post_per_batch():
    posts: list[bytes] = []
    runtime = ConsensusRuntime(
        PaxosNode(
            "a",
            MemoryStore(),
            ClusterStateMachine(),
            lambda to, msg: None,
            random.Random(1),
            PaxosConfig(),
            genesis={"a": "http://a", "b": "http://b"},
        ),
        MessageSigner(b"shared"),
        {"a": "http://a", "b": "http://b"},
    )
    runtime._post = lambda url, body, headers: posts.append(body)
    msgs = [m.Heartbeat("a", 0, Ballot(0, ""), n) for n in range(5)]
    runtime._deliver("b", msgs)
    assert len(posts) == 1 and len(json.loads(posts[0])) == 5
