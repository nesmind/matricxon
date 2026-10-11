"""Configuring the seed (the existing, populated node) for cluster mode through its own API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.cluster.configure import ConfigureRefusedError, ConfigureRequest, NodeConfigurator
from app.cluster.provisioning import EnvFile, InvalidProvisionError
from app.config import Settings
from app.dependencies import get_node_configurator
from app.main import MatricxonApp
from app.storage.paths import StoragePaths


def _configurator(tmp_path: Path, restarts: list, role: str = "off", **overrides: object):
    values = {
        "models_dir": tmp_path / "models",
        "prompt_cache_dir": tmp_path / "state" / "prompt_cache",
        "node_state_dir": tmp_path / "node",
        "cluster_role": role,
    } | overrides
    config = Settings(_env_file=None, **values)
    env = EnvFile(tmp_path / ".env")
    return NodeConfigurator(
        config, StoragePaths(config), env, lambda: restarts.append(1), delay=0.0
    ), env


def _shared(tmp_path: Path, key: bool = True) -> tuple[str, str]:
    (tmp_path / "shared" / "models").mkdir(parents=True)
    (tmp_path / "shared" / "cluster" / "prompt_cache").mkdir(parents=True)
    if key:
        (tmp_path / "shared" / "cluster" / "prompt_cache.key").write_bytes(b"k" * 32)
    return str(tmp_path / "shared" / "models"), str(
        tmp_path / "shared" / "cluster" / "prompt_cache"
    )


def _wait_restarts(restarts: list, count: int = 1) -> None:
    import time

    deadline = time.monotonic() + 3
    while len(restarts) < count and time.monotonic() < deadline:
        time.sleep(0.01)


def test_turning_cluster_mode_on_writes_the_settings_and_restarts_the_node(tmp_path: Path):
    restarts: list = []
    configurator, _ = _configurator(tmp_path, restarts)
    models, cache = _shared(tmp_path)
    request = ConfigureRequest("seed", models, cache, "http://seed:8420")
    assert configurator.configure(request) is True
    _wait_restarts(restarts)
    text = (tmp_path / ".env").read_text()
    assert "MATRICXON_CLUSTER_ROLE=seed" in text and f"MATRICXON_MODELS_DIR={models}" in text
    assert (
        f"MATRICXON_PROMPT_CACHE_DIR={cache}" in text
        and "MATRICXON_ADVERTISE_URL=http://seed:8420" in text
    )
    assert restarts == [1]


def test_nothing_changes_nothing_restarts(tmp_path: Path):
    models, cache = _shared(tmp_path)
    restarts: list = []
    configurator, _ = _configurator(
        tmp_path,
        restarts,
        role="seed",
        models_dir=Path(models),
        prompt_cache_dir=Path(cache),
        advertise_url="http://seed:8420",
        cache_key="k",
    )
    assert (
        configurator.configure(ConfigureRequest("seed", models, cache, "http://seed:8420")) is False
    )
    assert not (tmp_path / ".env").exists() and restarts == []


def test_turning_it_off_removes_the_role_and_keeps_the_other_lines(tmp_path: Path):
    restarts: list = []
    configurator, _ = _configurator(tmp_path, restarts, role="seed", cache_key="k")
    (tmp_path / ".env").write_text("MATRICXON_LOG_LEVEL=1\nMATRICXON_CLUSTER_ROLE=seed\n")
    assert configurator.configure(ConfigureRequest("off")) is True
    _wait_restarts(restarts)
    assert (tmp_path / ".env").read_text() == "MATRICXON_LOG_LEVEL=1\n"


def test_an_empty_string_clears_the_address(tmp_path: Path):
    restarts: list = []
    configurator, _ = _configurator(
        tmp_path, restarts, role="seed", cache_key="k", advertise_url="http://x:1"
    )
    (tmp_path / ".env").write_text(
        "MATRICXON_ADVERTISE_URL=http://x:1\nMATRICXON_CLUSTER_ROLE=seed\n"
    )
    assert configurator.configure(ConfigureRequest("seed", advertise_url="")) is True
    assert "ADVERTISE" not in (tmp_path / ".env").read_text()


def test_it_refuses_a_missing_folder_or_a_missing_cache_key_and_writes_nothing(tmp_path: Path):
    restarts: list = []
    configurator, _ = _configurator(tmp_path, restarts)
    with pytest.raises(InvalidProvisionError, match="existing absolute folder"):
        configurator.configure(ConfigureRequest("seed", models_dir="relative"))
    models, cache = _shared(tmp_path, key=False)
    with pytest.raises(InvalidProvisionError, match="cache key"):
        configurator.configure(ConfigureRequest("seed", models, cache))
    assert not (tmp_path / ".env").exists() and restarts == []


def test_a_joiner_cannot_be_configured_as_a_seed(tmp_path: Path):
    configurator, _ = _configurator(tmp_path, [], role="joiner")
    with pytest.raises(ConfigureRefusedError):
        configurator.configure(ConfigureRequest("seed"))


def test_the_endpoint_maps_the_outcomes(tmp_path: Path):
    restarts: list = []
    configurator, _ = _configurator(tmp_path, restarts)
    app = MatricxonApp().build()
    app.dependency_overrides[get_node_configurator] = lambda: configurator
    client = TestClient(app)
    models, cache = _shared(tmp_path)
    body = {
        "role": "seed",
        "models_dir": models,
        "prompt_cache_dir": cache,
        "advertise_url": "http://s:1",
    }
    assert client.post("/api/cluster/configure", json=body).json() == {"restarting": True}
    assert (
        client.post(
            "/api/cluster/configure", json={"role": "seed", "models_dir": "nope"}
        ).status_code
        == 400
    )
    refused, _ = _configurator(tmp_path, [], role="joiner")
    app.dependency_overrides[get_node_configurator] = lambda: refused
    assert client.post("/api/cluster/configure", json={"role": "off"}).status_code == 409
