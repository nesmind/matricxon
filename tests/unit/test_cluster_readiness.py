from pathlib import Path

from app.cluster.fs_probe import FilesystemProbe
from app.cluster.readiness import ACTION, OK, ClusterReadiness, ReadinessReport
from app.config import Settings
from app.storage.paths import StoragePaths

MOUNTS = (
    "20 1 8:1 / / rw - ext4 /dev/sda1 rw\n"
    "30 20 0:40 / {shared} rw - nfs4 srv:/x rw\n"
    "31 20 0:41 / {spaced} rw - cifs //s/y rw\n"
)


def _probe(tmp_path: Path) -> FilesystemProbe:
    file = tmp_path / "mountinfo"
    spaced = str(tmp_path / "with space").replace(" ", "\\040")
    file.write_text(MOUNTS.format(shared=tmp_path / "nfs", spaced=spaced))
    return FilesystemProbe(file)


def _node(
    tmp_path: Path,
    name: str,
    role: str,
    shared: Path | None = None,
    trust: bool = True,
    key: str | None = "k",
    probe: FilesystemProbe | None = None,
) -> ClusterReadiness:
    root = shared or tmp_path / "shared"
    settings = Settings(
        _env_file=None,
        models_dir=root / "models",
        prompt_cache_dir=root / "state" / "prompt_cache",
        node_state_dir=tmp_path / name,
        cluster_role=role,
        cluster_trust_shared_paths=trust,
        cache_key=key,
    )
    for sub in ("models", "state/prompt_cache"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return ClusterReadiness(settings, StoragePaths(settings), name, probe)


def _status(report: ReadinessReport) -> dict[str, str]:
    return {c.id: c.status for c in report.checks}


def test_probe_finds_the_mount_a_path_is_on(tmp_path: Path) -> None:
    probe = _probe(tmp_path)
    (tmp_path / "nfs" / "a").mkdir(parents=True)
    assert probe.fstype(tmp_path / "nfs" / "a") == "nfs4" and probe.is_shared(tmp_path / "nfs")
    (tmp_path / "with space").mkdir()
    assert probe.fstype(tmp_path / "with space") == "cifs"
    assert not probe.is_shared(tmp_path)


def test_local_paths_give_move_instructions_with_size_and_setting(tmp_path: Path) -> None:
    node = _node(tmp_path, "s", "seed", trust=False, probe=_probe(tmp_path))
    (tmp_path / "shared" / "state" / "prompt_cache" / "x.mxc").write_bytes(b"1" * 10)
    report = node.report()
    cache = next(c for c in report.checks if c.id == "prompt_cache")
    assert cache.status == ACTION and cache.size_bytes == 10
    assert cache.setting == "MATRICXON_PROMPT_CACHE_DIR" and "ext4" in cache.hint
    assert not report.ready


def test_a_shared_filesystem_passes(tmp_path: Path) -> None:
    (tmp_path / "nfs").mkdir()
    node = _node(
        tmp_path, "s", "seed", shared=tmp_path / "nfs", trust=False, probe=_probe(tmp_path)
    )
    assert _status(node.report())["models"] == OK


def test_missing_key_blocks_until_provisioned(tmp_path: Path) -> None:
    node = _node(tmp_path, "s", "seed", key=None)
    assert _status(node.report())["cache_key"] == ACTION
    (tmp_path / "shared" / "state" / "prompt_cache.key").write_bytes(b"k" * 32)
    assert _status(node.report())["cache_key"] == OK


def test_seed_activates_only_when_ready_and_claims_the_cluster(tmp_path: Path) -> None:
    blocked = _node(tmp_path, "s", "seed", key=None)
    assert not blocked.activate().active and not (tmp_path / "shared/state/.cluster").exists()
    seed = _node(tmp_path, "s", "seed")
    report = seed.activate()
    assert report.ready and report.active and report.nodes == ["s"]
    assert (tmp_path / "shared/state/.cluster/seed.json").is_file()


def test_off_role_reports_but_never_activates(tmp_path: Path) -> None:
    node = _node(tmp_path, "s", "off")
    assert node.report().ready and not node.activate().active


def test_joiner_needs_the_seed_then_joins(tmp_path: Path) -> None:
    joiner = _node(tmp_path, "j", "joiner")
    assert _status(joiner.report())["seed_visible"] == ACTION
    assert not joiner.activate().active
    _node(tmp_path, "s", "seed").activate()
    report = joiner.activate()
    assert report.ready and report.active and report.nodes == ["j", "s"]


def test_joiner_with_its_own_data_is_refused(tmp_path: Path) -> None:
    _node(tmp_path, "s", "seed").activate()
    joiner = _node(tmp_path, "j", "joiner")
    (tmp_path / "j").mkdir(exist_ok=True)
    (tmp_path / "j" / "node_id").write_text("j")
    assert _status(joiner.report())["node_fresh"] == OK
    (tmp_path / "j" / "old_chats.db").write_text("x")
    report = joiner.activate()
    assert _status(report)["node_fresh"] == ACTION and "old_chats.db" in report.checks[3].hint
    assert not report.active


def test_a_second_seed_cannot_claim_the_same_cluster(tmp_path: Path) -> None:
    _node(tmp_path, "s1", "seed").activate()
    second = _node(tmp_path, "s2", "seed")
    report = second.activate()
    assert _status(report)["single_seed"] == ACTION and not report.active
    assert _node(tmp_path, "s1", "seed").report().ready


def test_first_boot_leftovers_do_not_make_a_joiner_unfresh(tmp_path: Path) -> None:
    _node(tmp_path, "s", "seed").activate()
    joiner = _node(tmp_path, "j", "joiner")
    local = tmp_path / "j"
    (local / "prompt_cache").mkdir(parents=True)
    (local / "prompt_cache.key").write_bytes(b"k" * 32)
    (local / "node_id").write_text("j")
    assert _status(joiner.report())["node_fresh"] == OK


def _with_consensus(node: ClusterReadiness, running: bool) -> ClusterReadiness:
    node._consensus_running = lambda: running
    return node


def test_the_first_setup_checks_go_away_once_consensus_runs(tmp_path: Path) -> None:
    first_setup = {"seed_visible", "node_fresh", "single_seed"}
    seed = _node(tmp_path, "s", "seed")
    joiner = _node(tmp_path, "j", "joiner", shared=tmp_path / "shared")
    seed.activate()

    assert "single_seed" in _status(seed.report())
    assert {"seed_visible", "node_fresh"} <= set(_status(joiner.report()))

    for node in (seed, joiner):
        report = _with_consensus(node, True).report()
        assert not first_setup & set(_status(report))
        assert {"models", "prompt_cache", "cache_key"} <= set(_status(report))
        assert report.ready

    assert "single_seed" in _status(_with_consensus(seed, False).report())
