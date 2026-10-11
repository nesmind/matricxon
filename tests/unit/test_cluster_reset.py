"""Resetting a removed joiner to a fresh install: its setup and history go, shared folders stay."""

import time
from pathlib import Path

import pytest

from app.cluster.freshness import FreshnessChecker
from app.cluster.provisioning import EnvFile
from app.cluster.reset import CLUSTER_ENV_KEYS, NodeResetter, ResetRefusedError
from app.config import Settings
from app.storage.paths import StoragePaths


def _joiner(tmp_path: Path, role: str = "joiner", voter: bool = False):
    shared_models, shared_cache = tmp_path / "shared" / "models", tmp_path / "shared" / "cache"
    shared_models.mkdir(parents=True), shared_cache.mkdir(parents=True)
    (shared_models / "hf.co").mkdir()
    (shared_models / "hf.co" / "m.gguf.json").write_text("{}")
    (shared_cache / "chat.mxc").write_text("x")
    state = tmp_path / "state"
    state.mkdir()
    (state / "consensus.journal").write_text('{"k": "g"}\n')
    (state / "consensus.journal.bak-1").write_text("old")
    (state / "prompt_cache_policy.json").write_text("{}")
    (state / "node_id").write_text("abc")
    env = tmp_path / ".env"
    env.write_text(
        "MATRICXON_TORCH_THREADS=2\n"
        f"MATRICXON_CLUSTER_ROLE={role}\n"
        f"MATRICXON_MODELS_DIR={shared_models}\n"
        f"MATRICXON_PROMPT_CACHE_DIR={shared_cache}\n"
        "MATRICXON_ADVERTISE_URL=http://n:8420\n"
        "MATRICXON_CONSENSUS_SEEDS=a=http://s:8420\n"
    )
    settings = Settings(
        cluster_role=role,
        node_state_dir=state,
        models_dir=tmp_path / "own-models",
        prompt_cache_dir=tmp_path / "own-cache",
    )
    paths = StoragePaths(settings, pid_file=tmp_path / "pid")
    restarts: list[int] = []
    resetter = NodeResetter(
        settings,
        paths,
        EnvFile(env),
        lambda: voter,
        lambda: restarts.append(1),
        delay=0.0,
        own_cache_dir=tmp_path / "own-cache",
    )
    return resetter, settings, paths, env, restarts, (shared_models, shared_cache)


def test_a_removed_joiner_is_reset_and_the_shared_folders_are_left_alone(tmp_path):
    resetter, settings, paths, env, restarts, (models, cache) = _joiner(tmp_path)

    assert resetter.reset() is True
    time.sleep(0.2)

    assert restarts == [1]
    kept = env.read_text()
    assert "MATRICXON_TORCH_THREADS=2" in kept
    assert not any(key in kept for key in CLUSTER_ENV_KEYS)
    state = paths.consensus_journal.parent
    assert not list(state.glob("consensus.journal*")) and not paths.cache_policy_file.exists()
    assert not (state / "node_id").exists()
    assert (models / "hf.co" / "m.gguf.json").exists() and (cache / "chat.mxc").exists()


def test_after_the_reset_the_node_is_fresh_once_it_runs_with_its_defaults(tmp_path):
    resetter, settings, paths, env, _, _ = _joiner(tmp_path)
    resetter.reset()
    time.sleep(0.2)
    after = Settings(
        cluster_role="off",
        node_state_dir=settings.node_state_dir,
        models_dir=tmp_path / "own-models",
        prompt_cache_dir=tmp_path / "own-cache",
    )
    assert FreshnessChecker(after, StoragePaths(after, pid_file=tmp_path / "pid")).report().fresh


def test_the_first_node_can_be_reset_once_it_is_no_longer_a_voter(tmp_path):
    resetter, _, paths, env, restarts, _ = _joiner(tmp_path, role="seed")

    assert resetter.reset() is True
    time.sleep(0.2)

    assert "MATRICXON_CLUSTER_ROLE" not in env.read_text() and restarts == [1]
    assert not paths.consensus_journal.exists()


def test_a_hard_reset_needs_no_cluster_role_and_nothing_left_to_clean(tmp_path):
    """Reset is reset: even a node with no role and no leftovers is reset (and restarted)."""
    resetter, _, paths, _, restarts, _ = _joiner(tmp_path, role="off")
    for path in [
        *paths.consensus_journal.parent.glob("consensus.journal*"),
        paths.cache_policy_file,
    ]:
        path.unlink(missing_ok=True)
    assert resetter.reset() is True and restarts == [1]


def test_a_node_with_no_role_is_still_cleaned_of_what_it_kept_of_its_time_in_a_cluster(tmp_path):
    """a removed node, started plain, counted as not fresh and could not be reset."""
    resetter, settings, paths, _, restarts, _ = _joiner(tmp_path, role="off")
    (settings.prompt_cache_dir).mkdir()
    (settings.prompt_cache_dir / "old-chat.mxc").write_text("x")
    assert resetter.reset() is True and restarts == [1]
    assert not (settings.prompt_cache_dir / "old-chat.mxc").exists()
    assert not paths.consensus_journal.exists() and not paths.cache_policy_file.exists()


def test_a_shared_cache_folder_is_never_cleaned(tmp_path):
    resetter, settings, paths, _, _, (_, shared_cache) = _joiner(tmp_path)
    settings.prompt_cache_dir = shared_cache
    resetter.reset()
    assert (shared_cache / "chat.mxc").exists()


def test_a_node_that_is_still_a_voter_is_refused(tmp_path):
    resetter, _, paths, env, restarts, _ = _joiner(tmp_path, voter=True)
    with pytest.raises(ResetRefusedError, match="still a voter"):
        resetter.reset()
    assert paths.consensus_journal.exists() and restarts == []


def test_a_forced_reset_also_resets_a_voter(tmp_path):
    resetter, _, paths, env, restarts, _ = _joiner(tmp_path, voter=True)
    assert resetter.reset(force=True) is True
    assert not paths.consensus_journal.exists()


def _markers(paths, own="abc", seed="abc"):
    nodes = paths.cluster_dir / "nodes"
    nodes.mkdir(parents=True)
    for name in (own, "other"):
        (nodes / f"{name}.json").write_text("{}")
    (paths.cluster_dir / "seed.json").write_text(f'{{"node_id": "{seed}"}}')
    return nodes


def test_a_reset_clears_its_own_markers_and_a_seed_claim_that_is_its_own(tmp_path):
    resetter, _, paths, *_ = _joiner(tmp_path, role="seed")
    nodes = _markers(paths)
    resetter.reset()
    assert not (nodes / "abc.json").exists() and (nodes / "other.json").exists()
    assert not (paths.cluster_dir / "seed.json").exists() and not paths.node_id_file.exists()


def test_a_reset_leaves_another_nodes_seed_claim(tmp_path):
    resetter, _, paths, *_ = _joiner(tmp_path)
    _markers(paths, seed="someone-else")
    resetter.reset()
    assert (paths.cluster_dir / "seed.json").exists()


def test_a_forced_reset_clears_every_marker(tmp_path):
    resetter, _, paths, *_ = _joiner(tmp_path, voter=True)
    nodes = _markers(paths, seed="someone-else")
    resetter.reset(force=True)
    assert not (paths.cluster_dir / "seed.json").exists() and not list(nodes.glob("*.json"))
