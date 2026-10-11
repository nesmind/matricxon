from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cluster.freshness import FreshnessChecker
from app.cluster.provisioning import (
    EnvFile,
    InvalidProvisionError,
    NodeProvisioner,
    NotFreshError,
    ProvisionRequest,
)
from app.config import Settings
from app.dependencies import get_freshness_checker, get_node_provisioner
from app.main import MatricxonApp
from app.storage.paths import StoragePaths


def _node(tmp_path: Path, role: str = "off") -> tuple[FreshnessChecker, StoragePaths]:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        prompt_cache_dir=tmp_path / "state" / "prompt_cache",
        node_state_dir=tmp_path / "node",
        cluster_role=role,
    )
    paths = StoragePaths(settings)
    return FreshnessChecker(settings, paths), paths


def _shared(tmp_path: Path) -> ProvisionRequest:
    (tmp_path / "shared" / "models").mkdir(parents=True)
    (tmp_path / "shared" / "cluster" / "prompt_cache").mkdir(parents=True)
    (tmp_path / "shared" / "cluster" / "prompt_cache.key").write_bytes(b"k" * 32)
    return ProvisionRequest(
        str(tmp_path / "shared" / "models"),
        str(tmp_path / "shared" / "cluster" / "prompt_cache"),
        False,
    )


def test_an_empty_install_is_fresh_even_with_its_own_auto_created_key(tmp_path: Path) -> None:
    checker, paths = _node(tmp_path)
    (tmp_path / "node").mkdir()
    (tmp_path / "node" / "node_id").write_text("n")
    paths.cache_key_file.parent.mkdir(parents=True)
    paths.cache_key_file.write_bytes(b"k" * 32)
    assert checker.report().fresh


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (
            lambda p: (
                (p.models_dir / "a").mkdir(parents=True)
                or (p.models_dir / "a/m.gguf.json").write_text("{}")
            ),
            "installed model",
        ),
        (
            lambda p: (
                p.prompt_cache_dir.mkdir(parents=True)
                or (p.prompt_cache_dir / "x.mxc").write_bytes(b"1")
            ),
            "chat cache",
        ),
        (
            lambda p: (
                p.cache_policy_file.parent.mkdir(parents=True)
                or p.cache_policy_file.write_text("{}")
            ),
            "cache policy",
        ),
    ],
)
def test_a_populated_install_is_not_fresh(tmp_path: Path, setup, reason: str) -> None:
    checker, paths = _node(tmp_path)
    setup(paths)
    report = checker.report()
    assert not report.fresh and any(reason in r for r in report.reasons)


def test_other_local_state_and_an_existing_cluster_role_are_not_fresh(tmp_path: Path) -> None:
    checker, _ = _node(tmp_path)
    (tmp_path / "node").mkdir()
    (tmp_path / "node" / "chats.db").write_text("x")
    assert "chats.db" in checker.report().reasons[0]
    seed, _ = _node(tmp_path, "seed")
    assert any("cluster role 'seed'" in r for r in seed.report().reasons)


def test_env_file_updates_its_keys_and_keeps_the_rest(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# note\nMATRICXON_LOG_LEVEL=1\nMATRICXON_CLUSTER_ROLE=off\n")
    EnvFile(env).set({"MATRICXON_CLUSTER_ROLE": "joiner", "MATRICXON_MODELS_DIR": "/m"})
    assert env.read_text().splitlines() == [
        "# note",
        "MATRICXON_LOG_LEVEL=1",
        "MATRICXON_CLUSTER_ROLE=joiner",
        "MATRICXON_MODELS_DIR=/m",
    ]


def test_provision_writes_the_joiner_settings_and_restarts_when_asked(tmp_path: Path) -> None:
    checker, _ = _node(tmp_path)
    killed: list[bool] = []
    request = _shared(tmp_path)
    env = EnvFile(tmp_path / ".env")
    provisioner = NodeProvisioner(checker, env, kill=lambda: killed.append(True))

    assert provisioner.provision(request) is False and not killed
    text = (tmp_path / ".env").read_text()
    assert "MATRICXON_CLUSTER_ROLE=joiner" in text
    assert f"MATRICXON_MODELS_DIR={request.models_dir}" in text
    assert f"MATRICXON_PROMPT_CACHE_DIR={request.prompt_cache_dir}" in text
    assert "MATRICXON_CACHE_KEY" not in text


def test_provision_refuses_a_node_that_is_not_fresh_and_writes_nothing(tmp_path: Path) -> None:
    checker, paths = _node(tmp_path)
    paths.prompt_cache_dir.mkdir(parents=True)
    (paths.prompt_cache_dir / "x.mxc").write_bytes(b"1")
    with pytest.raises(NotFreshError, match="chat cache"):
        NodeProvisioner(checker, EnvFile(tmp_path / ".env")).provision(_shared(tmp_path))
    assert not (tmp_path / ".env").exists()


def test_provision_rejects_unusable_folders_and_a_missing_shared_key(tmp_path: Path) -> None:
    checker, _ = _node(tmp_path)
    provisioner = NodeProvisioner(checker, EnvFile(tmp_path / ".env"))
    with pytest.raises(InvalidProvisionError, match="models_dir"):
        provisioner.provision(ProvisionRequest("relative", "/tmp", False))
    request = _shared(tmp_path)
    (tmp_path / "shared" / "cluster" / "prompt_cache.key").unlink()
    with pytest.raises(InvalidProvisionError, match="shared cache key"):
        provisioner.provision(request)
    assert not (tmp_path / ".env").exists()


def test_endpoints_report_freshness_and_map_errors(tmp_path: Path) -> None:
    checker, _ = _node(tmp_path)
    app = MatricxonApp().build()
    app.dependency_overrides[get_freshness_checker] = lambda: checker
    app.dependency_overrides[get_node_provisioner] = lambda: NodeProvisioner(
        checker, EnvFile(tmp_path / ".env"), kill=lambda: None
    )
    client = TestClient(app)
    assert client.get("/api/cluster/fresh").json() == {"fresh": True, "reasons": []}
    bad = client.post("/api/cluster/provision", json={"models_dir": "x", "prompt_cache_dir": "y"})
    assert bad.status_code == 400
    request = _shared(tmp_path)
    ok = client.post(
        "/api/cluster/provision",
        json={
            "models_dir": request.models_dir,
            "prompt_cache_dir": request.prompt_cache_dir,
            "restart": False,
        },
    )
    assert ok.status_code == 200 and ok.json() == {"restarting": False}
    # the written.env isn't loaded into this test's settings, so the node still reports fresh; a
    # populated node is refused with 409
    (tmp_path / "node").mkdir()
    (tmp_path / "node" / "chats.db").write_text("x")
    refused = client.post(
        "/api/cluster/provision",
        json={"models_dir": request.models_dir, "prompt_cache_dir": request.prompt_cache_dir},
    )
    assert refused.status_code == 409 and "not a fresh install" in refused.json()["detail"]


def test_a_restart_in_place_must_not_resurrect_a_setting_the_env_file_dropped(tmp_path):
    """pAIring starts the seed with the cluster role in the process environment; execv keeps it and
    a real environment
    variable beats.env, so cluster mode could never be turned off.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("MATRICXON_CLUSTER_ROLE=seed\nOTHER=1\n")
    role, models = "MATRICXON_CLUSTER_ROLE", "MATRICXON_MODELS_DIR"
    environ = {role: "seed", models: "/shared/m", "UNRELATED": "x"}
    env = EnvFile(env_file, environ)

    env.remove([role])
    assert role not in environ and role not in env_file.read_text()
    env.set({models: "/new/m"})
    assert environ[models] == "/new/m"
    assert f"{models}=/new/m" in env_file.read_text()
    assert environ["UNRELATED"] == "x" and "OTHER=1" in env_file.read_text()


def test_without_an_environ_the_env_file_alone_changes(tmp_path):
    env_file = tmp_path / ".env"
    EnvFile(env_file).set({"A": "1"})
    EnvFile(env_file).remove(["A"])
    assert env_file.read_text() == ""
