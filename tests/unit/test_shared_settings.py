"""Folders and timers kept in the replicated cluster config, applied by each node itself."""

from pathlib import Path

from app.cluster.provisioning import EnvFile
from app.cluster.shared_settings import CACHE_KEY, PATHS_KEY, TIMERS_KEY, SharedSettings
from app.config import Settings
from app.consensus.timers import apply_timers, valid_timers
from tests.consensus.sim import SimCluster


class FakeRuntime:
    """The three things SharedSettings uses of a ConsensusRuntime."""

    def __init__(self, node, config: dict) -> None:
        self.node, self.config, self.note = node, config, None

    def shared_config(self) -> dict:
        return self.config

    def retune(self, election: float, heartbeat: float) -> None:
        apply_timers(self.node, election, heartbeat)


def _node():
    cluster = SimCluster(seed=1)
    cluster.run(0.1)
    return cluster.nodes["n0"]


def _shared(tmp_path: Path, config: dict, **settings):
    models, cache = tmp_path / "m", tmp_path / "c"
    models.mkdir(), cache.mkdir()
    restarts: list[int] = []
    conf = Settings(models_dir=models, prompt_cache_dir=cache, **settings)
    env = EnvFile(tmp_path / ".env")
    runtime = FakeRuntime(_node(), config)
    shared = SharedSettings(runtime, conf, env, lambda: restarts.append(1), delay=(0.0, 0.0))
    return shared, runtime, restarts, tmp_path


def test_timers_must_make_sense():
    assert valid_timers({"election_seconds": 5, "heartbeat_seconds": 1}) == (5.0, 1.0)
    assert valid_timers({"election_seconds": 1, "heartbeat_seconds": 0.5}) is None
    assert valid_timers({"election_seconds": 0.1, "heartbeat_seconds": 0.01}) is None
    assert valid_timers({"election_seconds": 100, "heartbeat_seconds": 1}) is None
    assert valid_timers({"election_seconds": "x"}) is None and valid_timers(None) is None


def test_replicated_timers_are_applied_live_and_garbage_is_ignored(tmp_path):
    shared, runtime, restarts, _ = _shared(
        tmp_path, {TIMERS_KEY: {"election_seconds": 5, "heartbeat_seconds": 1}}
    )
    shared.tick()
    config = runtime.node.config
    assert (config.election_timeout, config.heartbeat, config.retry_interval) == (5, 1, 2.5)
    assert runtime.node._watch._timeout == 5 and not restarts
    runtime.config[TIMERS_KEY] = {"election_seconds": 1, "heartbeat_seconds": 9}
    shared.tick()
    assert runtime.node.config.election_timeout == 5


def test_a_node_adopts_the_cluster_folders_and_restarts_once(tmp_path):
    new = tmp_path / "new"
    new.mkdir()
    shared, runtime, restarts, _ = _shared(
        tmp_path, {PATHS_KEY: {"models_dir": str(new), "prompt_cache_dir": ""}}
    )
    shared.tick()
    shared.tick()
    import time

    time.sleep(0.2)
    assert (tmp_path / ".env").read_text().strip() == f"MATRICXON_MODELS_DIR={new}"
    assert restarts == [1]
    assert "restarting" in runtime.note


def test_folders_this_node_does_not_have_are_left_alone(tmp_path):
    shared, runtime, restarts, _ = _shared(tmp_path, {PATHS_KEY: {"models_dir": "/nope/not/here"}})
    shared.tick()
    assert not (tmp_path / ".env").exists() and not restarts
    assert "/nope/not/here" in runtime.note


def test_the_same_folders_change_nothing(tmp_path):
    models, cache = tmp_path / "m", tmp_path / "c"
    shared, runtime, restarts, _ = _shared(
        tmp_path, {PATHS_KEY: {"models_dir": str(models) + "/", "prompt_cache_dir": str(cache)}}
    )
    shared.tick()
    assert not (tmp_path / ".env").exists() and not restarts and runtime.note is None


class FakePolicy:
    def __init__(self) -> None:
        self.enabled, self.budget_mb, self.ttl_hours = True, 2048, 168

    def update(self, enabled: bool, budget_mb: int, ttl_hours: int) -> None:
        self.enabled, self.budget_mb, self.ttl_hours = enabled, budget_mb, ttl_hours


class FakeStore:
    def __init__(self) -> None:
        self.policy, self.cleared, self.enforced = FakePolicy(), 0, 0

    def clear(self) -> None:
        self.cleared += 1

    def enforce_limits(self) -> None:
        self.enforced += 1


def _with_cache(tmp_path: Path, value):
    shared, runtime, _, _ = _shared(tmp_path, {CACHE_KEY: value})
    store = FakeStore()
    shared._cache = lambda: store
    return shared, store


def test_replicated_cache_limits_are_applied_once(tmp_path):
    shared, store = _with_cache(tmp_path, {"enabled": True, "budget_mb": 512, "ttl_hours": 24})
    shared.tick()
    shared.tick()
    assert (store.policy.budget_mb, store.policy.ttl_hours) == (512, 24)
    assert store.enforced == 1 and store.cleared == 0


def test_turning_the_cache_off_through_the_cluster_deletes_what_was_saved(tmp_path):
    shared, store = _with_cache(tmp_path, {"enabled": False, "budget_mb": 2048, "ttl_hours": 168})
    shared.tick()
    assert store.policy.enabled is False and store.cleared == 1


def test_nonsense_cache_limits_are_ignored(tmp_path):
    bad_values = (
        None,
        {},
        {"enabled": "yes", "budget_mb": 1, "ttl_hours": 1},
        {"enabled": True, "budget_mb": -1, "ttl_hours": 1},
    )
    for i, bad in enumerate(bad_values):
        (tmp_path / str(i)).mkdir()
        shared, store = _with_cache(tmp_path / str(i), bad)
        shared.tick()
        assert store.enforced == 0 and store.policy.budget_mb == 2048
