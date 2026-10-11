import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings
from app.consensus import messages as m
from app.consensus.auth import MessageSigner
from app.consensus.codec import encode
from app.consensus.factory import ConsensusConfigError, ConsensusFactory
from app.consensus.runtime import SIGNATURE_HEADER
from app.dependencies import get_consensus_runtime
from app.main import MatricxonApp


def _factory(tmp_path: Path, **settings: object) -> ConsensusFactory:
    config = Settings(_env_file=None, **settings)
    return ConsensusFactory(config, "me", tmp_path / "node" / "consensus.journal", tmp_path / "key")


def test_genesis_is_parsed_from_the_setting():
    assert ConsensusFactory.parse_genesis("a=http://h1:8420, b=http://h2:8420/ ,") == {
        "a": "http://h1:8420",
        "b": "http://h2:8420",
    }
    assert ConsensusFactory.parse_genesis("") == {}
    for bad in ("a", "=http://x", "a="):
        with pytest.raises(ConsensusConfigError):
            ConsensusFactory.parse_genesis(bad)


def test_consensus_is_off_without_genesis(tmp_path: Path):
    assert _factory(tmp_path).build() is None


def test_a_node_must_be_one_of_the_first_voters_and_needs_a_secret(tmp_path: Path):
    with pytest.raises(ConsensusConfigError, match="is not in"):
        _factory(tmp_path, consensus_genesis="a=http://a", cache_key="k").build()
    with pytest.raises(ConsensusConfigError, match="CONSENSUS_SECRET"):
        _factory(tmp_path, consensus_genesis="me=http://me").build()


def test_the_secret_comes_from_the_setting_then_the_cache_key_then_the_key_file(tmp_path: Path):
    def signature(factory: ConsensusFactory) -> str:
        return factory.build()._signer.sign(b"x")

    explicit = _factory(
        tmp_path, consensus_genesis="me=http://me", consensus_secret="s", cache_key="k"
    )
    assert signature(explicit) == MessageSigner(b"s").sign(b"x")
    cache = _factory(tmp_path, consensus_genesis="me=http://me", cache_key="k")
    assert signature(cache) == MessageSigner(b"k").sign(b"x")
    (tmp_path / "key").write_bytes(b"filekey")
    assert signature(_factory(tmp_path, consensus_genesis="me=http://me")) == MessageSigner(
        b"filekey"
    ).sign(b"x")


def test_a_single_voter_cluster_leads_itself_and_keeps_its_journal(tmp_path: Path):
    runtime = _factory(tmp_path, consensus_genesis="me=http://me", cache_key="k").build()
    runtime.start()
    try:
        deadline = time.monotonic() + 5
        while runtime.status()["role"] != "leader" and time.monotonic() < deadline:
            time.sleep(0.02)
        _, applied, result = runtime.submit({"op": "set", "key": "a", "value": 1})
        assert applied and result is True
    finally:
        runtime.stop()
    assert (tmp_path / "node" / "consensus.journal").stat().st_size > 0
    again = _factory(tmp_path, consensus_genesis="me=http://me", cache_key="k").build()
    assert again.status()["config"] == {"a": 1}


@pytest.fixture
def client_with(tmp_path: Path):
    def make(runtime) -> TestClient:
        app = MatricxonApp().build()
        app.dependency_overrides[get_consensus_runtime] = lambda: runtime
        return TestClient(app)

    return make


def test_the_endpoints_say_consensus_is_off_when_it_is(client_with):
    client = client_with(None)
    assert client.get("/api/consensus/status").json()["enabled"] is False
    assert client.post("/api/consensus/message", content=b"{}").status_code == 503
    assert client.put("/api/consensus/config", json={"key": "k", "value": 1}).status_code == 503


def test_the_message_endpoint_checks_signature_and_format_and_the_config_endpoint_commits(
    tmp_path, client_with
):
    runtime = _factory(tmp_path, consensus_genesis="me=http://me", consensus_secret="s").build()
    runtime.start()
    try:
        client = client_with(runtime)
        body = json.dumps(encode(m.CatchUp("peer", 0, 0))).encode()
        good = MessageSigner(b"s").sign(body)
        assert (
            client.post(
                "/api/consensus/message", content=body, headers={SIGNATURE_HEADER: good}
            ).status_code
            == 204
        )
        assert client.post("/api/consensus/message", content=body).status_code == 401
        assert (
            client.post(
                "/api/consensus/message", content=body, headers={SIGNATURE_HEADER: "00"}
            ).status_code
            == 401
        )
        junk = b'{"type": "Nope"}'
        signed = {SIGNATURE_HEADER: MessageSigner(b"s").sign(junk)}
        assert (
            client.post("/api/consensus/message", content=junk, headers=signed).status_code == 400
        )

        deadline = time.monotonic() + 5
        while runtime.status()["role"] != "leader" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert client.put(
            "/api/consensus/config", json={"key": "keep_alive", "value": 600}
        ).json() == {
            "applied": True,
            "result": True,
        }
        status = client.get("/api/consensus/status").json()
        assert (
            status["enabled"]
            and status["role"] == "leader"
            and status["config"] == {"keep_alive": 600}
        )
        assert status["voters"] == ["me"] and status["epoch"] == 0
    finally:
        runtime.stop()


def test_the_server_starts_and_stops_the_runtime_with_the_app(monkeypatch, tmp_path):
    # never the real server's pid file
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    events: list[str] = []
    runtime = SimpleNamespace(
        start=lambda: events.append("start"), stop=lambda: events.append("stop")
    )
    monkeypatch.setattr(main, "get_consensus_runtime", lambda: runtime)
    monkeypatch.setattr(main, "get_cache_store", lambda: None)
    monkeypatch.setattr(main, "get_model_manager", lambda: SimpleNamespace(shutdown=lambda: None))
    app = MatricxonApp().build()

    async def go() -> None:
        async with app.router.lifespan_context(app):
            events.append("serving")

    asyncio.run(go())
    assert events == ["start", "serving", "stop"]


def test_a_node_with_cluster_history_keeps_running_consensus_without_its_address(
    tmp_path: Path,
) -> None:
    """Bug: saving the settings form wiped the advertise address; the restarted
    seed ran without
    consensus while still a voter and the cluster lost its quorum.
    """
    journal = tmp_path / "node" / "consensus.journal"
    secret = {"consensus_secret": "shared"}
    first = _factory(
        tmp_path, cluster_role="seed", advertise_url="http://me:8420", **secret
    ).build()
    assert first is not None
    assert journal.is_file() and journal.stat().st_size > 0

    again = _factory(tmp_path, cluster_role="seed", **secret).build()
    assert again is not None and again.node.members == ["me"]
    assert _factory(tmp_path, cluster_role="joiner", **secret).build() is not None
    assert _factory(tmp_path, cluster_role="off", **secret).build() is None


def test_a_seed_without_an_advertise_address_uses_its_own_address(
    tmp_path: Path, monkeypatch
) -> None:
    """Setting up a cluster needs no typed address: the machine's own is the default."""
    from app.consensus import factory

    monkeypatch.setattr(factory, "own_address", lambda port: f"http://10.1.2.3:{port}")
    runtime = _factory(tmp_path, cluster_role="seed", consensus_secret="shared").build()
    assert runtime is not None
    assert runtime.node.replica.members == {"me": "http://10.1.2.3:8420"}

    set_one = {"cluster_role": "seed", "advertise_url": "http://real:1", "consensus_secret": "s"}
    assert _factory(tmp_path / "b", **set_one).build().node.replica.members == {
        "me": "http://real:1"
    }


def test_a_seed_with_no_address_at_all_does_not_start_consensus(
    tmp_path: Path, monkeypatch
) -> None:
    from app.consensus import factory

    monkeypatch.setattr(factory, "own_address", lambda port: None)
    assert _factory(tmp_path, cluster_role="seed", consensus_secret="shared").build() is None
