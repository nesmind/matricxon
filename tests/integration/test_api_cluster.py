from pathlib import Path

from fastapi.testclient import TestClient

from app.cluster.readiness import ClusterReadiness
from app.config import Settings
from app.dependencies import get_cluster_readiness
from app.main import MatricxonApp
from app.storage.paths import StoragePaths


def _client(tmp_path: Path, role: str) -> TestClient:
    for sub in ("models", "state/prompt_cache"):
        (tmp_path / sub).mkdir(parents=True)
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        prompt_cache_dir=tmp_path / "state" / "prompt_cache",
        node_state_dir=tmp_path / "n",
        cluster_role=role,
        cluster_trust_shared_paths=True,
        cache_key="k",
    )
    readiness = ClusterReadiness(settings, StoragePaths(settings), "node-1")
    app = MatricxonApp().build()
    app.dependency_overrides[get_cluster_readiness] = lambda: readiness
    return TestClient(app)


def test_readiness_lists_checks_with_status_and_hints(tmp_path: Path) -> None:
    body = _client(tmp_path, "seed").get("/api/cluster/readiness").json()
    assert body["role"] == "seed" and body["ready"] is True and body["active"] is False
    assert {c["id"] for c in body["checks"]} >= {"models", "prompt_cache", "cache_key"}
    assert all({"what", "why", "hint", "size_bytes"} <= c.keys() for c in body["checks"])


def test_activate_announces_the_node(tmp_path: Path) -> None:
    client = _client(tmp_path, "seed")
    body = client.post("/api/cluster/activate").json()
    assert body["active"] is True and body["nodes"] == ["node-1"]
    assert client.get("/api/cluster/readiness").json()["active"] is True


def test_a_node_says_whether_it_can_reach_an_address(client, monkeypatch):
    import httpx

    from app.routers import consensus_router

    class Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"node_id": "n9"}

    def fake_get(url, timeout=None):
        if "down" in url:
            raise httpx.ConnectError("refused")
        return Reply()

    monkeypatch.setattr(consensus_router.httpx, "get", fake_get)
    assert (
        client.post("/api/consensus/reach", json={"address": "http://up:8420/"}).json()["reachable"]
        is True
    )
    down = client.post("/api/consensus/reach", json={"address": "http://down:8420"}).json()
    assert down["reachable"] is False and "refused" in down["detail"]
    assert (
        client.post("/api/consensus/reach", json={"address": "ftp://x"}).json()["reachable"]
        is False
    )
