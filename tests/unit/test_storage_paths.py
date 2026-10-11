from pathlib import Path

from app.config import Settings
from app.storage.node_identity import NodeIdentity
from app.storage.paths import StorageInspector, StoragePaths, StorageRole


def _paths(tmp_path: Path) -> StoragePaths:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "models",
        prompt_cache_dir=tmp_path / "state" / "prompt_cache",
        node_state_dir=tmp_path / "node",
    )
    return StoragePaths(settings, pid_file=tmp_path / "run.pid")


def test_state_files_live_next_to_the_cache_dir(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    assert paths.cache_key_file == tmp_path / "state" / "prompt_cache.key"
    assert paths.cache_policy_file == tmp_path / "state" / "prompt_cache_policy.json"
    assert paths.node_id_file == tmp_path / "node" / "node_id"


def test_entries_classify_every_path(tmp_path: Path) -> None:
    roles = {e.name: e.role for e in _paths(tmp_path).entries()}
    assert roles["models"] == StorageRole.SHARED
    assert roles["prompt_cache"] == StorageRole.SHARED
    assert roles["cache_key"] == StorageRole.REPLICATED
    assert roles["cache_policy"] == StorageRole.CLUSTER_CONFIG
    assert roles["native_build"] == roles["log"] == roles["pid"] == StorageRole.PER_NODE


def test_inspector_sizes_a_tree_and_reports_free_space_of_a_missing_path(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a").write_bytes(b"x" * 10)
    (tmp_path / "d" / "b").write_bytes(b"y" * 5)
    inspector = StorageInspector()
    usage = inspector.usage(tmp_path / "d")
    assert (usage.exists, usage.size_bytes) == (True, 15)
    missing = inspector.usage(tmp_path / "nope" / "deeper")
    assert (missing.exists, missing.size_bytes) == (False, 0)
    assert missing.free_bytes and missing.free_bytes > 0


def test_node_id_is_created_once_and_stable(tmp_path: Path) -> None:
    identity = NodeIdentity(tmp_path / "s" / "node_id")
    first = identity.load()
    assert first and identity.load() == first
    assert NodeIdentity(tmp_path / "s" / "node_id").load() == first
