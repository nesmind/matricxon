"""The cluster using consensus: it starts from a role and an address, a joiner is given its voters,
and leases decide who cleans the cache and who pulls models."""

import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.cluster.freshness import FreshnessChecker
from app.cluster.provisioning import EnvFile, NodeProvisioner, ProvisionRequest
from app.config import Settings
from app.consensus.factory import ConsensusConfigError, ConsensusFactory
from app.consensus.lease import LeaseKeeper, Leases
from app.dependencies import get_consensus_runtime
from app.main import MatricxonApp
from app.routers.pull_router import require_pull_lease
from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.server.errors import NotLeaseHolderError
from app.storage.paths import StoragePaths
from tests.unit.test_cache_store import _slot


def _factory(tmp_path: Path, **settings: object) -> ConsensusFactory:
    config = Settings(_env_file=None, cache_key="k", node_state_dir=tmp_path / "node", **settings)
    return ConsensusFactory(config, "me", tmp_path / "node" / "consensus.journal", tmp_path / "key")


def test_consensus_stays_off_without_a_role_and_an_address(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.consensus.factory.own_address", lambda port: None)
    assert _factory(tmp_path).build() is None
    assert _factory(tmp_path, cluster_role="seed").build() is None
    assert _factory(tmp_path, cluster_role="joiner").build() is None
    assert _factory(tmp_path, advertise_url="http://me:8420").build() is None


def test_a_seed_with_an_address_starts_a_group_of_one_and_claims_the_leases(tmp_path: Path):
    runtime = _factory(tmp_path, cluster_role="seed", advertise_url="http://me:8420/").build()
    assert runtime.node.members == ["me"] and runtime.node.replica.members == {
        "me": "http://me:8420"
    }
    assert runtime.keeper.names == ("model_pull", "cache_clean")
    runtime.start()
    try:
        deadline = time.monotonic() + 8
        while (
            set(runtime.status()["holds"]) != {"model_pull", "cache_clean"}
            and time.monotonic() < deadline
        ):
            time.sleep(0.05)
        assert set(runtime.status()["holds"]) == {
            "model_pull",
            "cache_clean",
        }
    finally:
        runtime.stop()


def test_a_joiner_does_not_vote_until_it_is_made_one_and_knows_where_to_catch_up(tmp_path: Path):
    runtime = _factory(
        tmp_path,
        cluster_role="joiner",
        consensus_seeds="seed1=http://seed:8420, seed2=http://s2:8420",
    ).build()
    assert not runtime.node.is_voter and runtime.node.members == []
    assert runtime.address_of("seed1") == "http://seed:8420"
    assert runtime.keeper is not None


def test_an_explicit_genesis_still_wins_and_a_node_must_be_in_its_group(tmp_path: Path):
    runtime = _factory(
        tmp_path,
        consensus_genesis="me=http://me,b=http://b",
        cluster_role="joiner",
        advertise_url="x",
    ).build()
    assert runtime.node.members == ["b", "me"]
    with pytest.raises(ConsensusConfigError):
        _factory(tmp_path, consensus_genesis="other=http://o").build()


def _node(tmp_path: Path, journal: bool = False) -> FreshnessChecker:
    settings = Settings(
        _env_file=None,
        models_dir=tmp_path / "m",
        prompt_cache_dir=tmp_path / "s" / "pc",
        node_state_dir=tmp_path / "n",
    )
    (tmp_path / "n").mkdir(exist_ok=True)
    if journal:
        (tmp_path / "n" / "consensus.journal").write_text("x")
    return FreshnessChecker(settings, StoragePaths(settings))


def test_a_consensus_journal_is_history_before_joining_but_normal_afterwards(tmp_path: Path):
    checker = _node(tmp_path, journal=True)
    assert "consensus.journal" in checker.report().reasons[0]
    assert checker.local_state() == []


def test_provisioning_hands_a_joiner_its_address_and_the_voters_to_catch_up_from(tmp_path: Path):
    checker = _node(tmp_path)
    for folder in ("models", "cluster/prompt_cache"):
        (tmp_path / "shared" / folder).mkdir(parents=True)
    (tmp_path / "shared" / "cluster" / "prompt_cache.key").write_bytes(b"k" * 32)
    request = ProvisionRequest(
        str(tmp_path / "shared" / "models"),
        str(tmp_path / "shared" / "cluster" / "prompt_cache"),
        restart=False,
        advertise_url="http://joiner:8420",
        consensus_seeds={"seed": "http://seed:8420", "two": "http://two:8420"},
    )
    NodeProvisioner(checker, EnvFile(tmp_path / ".env")).provision(request)
    env = (tmp_path / ".env").read_text()
    assert "MATRICXON_ADVERTISE_URL=http://joiner:8420" in env
    assert "MATRICXON_CONSENSUS_SEEDS=seed=http://seed:8420,two=http://two:8420" in env


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _store(tmp_path: Path) -> PersistentCacheStore:
    policy = PersistencePolicy(True, 100, 24)
    return PersistentCacheStore(tmp_path / "c", CacheCipher(bytes(range(32))), policy, "a")


def test_only_the_cleanup_owner_deletes_by_age_or_budget(tmp_path: Path):
    store = _store(tmp_path)
    store.spill_async("m|1", _slot(list(range(24))))
    store.flush()
    store.policy.ttl_hours = 1
    for entry in store._index.values():
        entry.saved_at -= 7200
    store.may_clean = lambda: False
    store.enforce_limits()
    assert store.usage()[0] == 1
    store.may_clean = lambda: True
    store.enforce_limits()
    assert store.usage()[0] == 0


def test_the_keeper_runs_a_job_only_while_it_owns_its_lease():
    class Clock:
        now = 100.0

        def __call__(self) -> float:
            return self.now

    config: dict = {}
    clock = Clock()

    def submit(payload, wait):
        from app.consensus.state_machine import ClusterStateMachine

        machine = ClusterStateMachine()
        machine.config = config
        return None, True, machine.apply(payload)

    mine = Leases("a", submit, lambda: config, clock)
    other = Leases("b", submit, lambda: config, clock)
    runs: list[int] = []
    keeper = LeaseKeeper(mine, ("clean",), ttl=9)
    keeper.add_task("clean", lambda: runs.append(1))
    other.try_acquire("clean", 9)
    keeper.tick()
    assert runs == []
    clock.now += 20
    keeper.tick()
    assert runs == [1]


def test_pulls_are_refused_on_a_node_without_the_pull_lease_and_say_who_has_it():
    owner = SimpleNamespace(holds=lambda name: True)
    other = SimpleNamespace(holds=lambda name: False)
    runtime = lambda keeper, holder: SimpleNamespace(  # noqa: E731
        keeper=keeper, leases=SimpleNamespace(holder=lambda name: holder)
    )
    require_pull_lease(None)
    require_pull_lease(runtime(owner, "me"))
    with pytest.raises(NotLeaseHolderError, match="node seed2 owns the model-pull lease"):
        require_pull_lease(runtime(other, "seed2"))
    with pytest.raises(NotLeaseHolderError, match="no node owns the pull lease yet"):
        require_pull_lease(runtime(other, None))


def test_the_pull_endpoint_answers_409_with_the_reason(monkeypatch):
    app = MatricxonApp().build()
    runtime = SimpleNamespace(
        keeper=SimpleNamespace(holds=lambda name: False),
        leases=SimpleNamespace(holder=lambda n: "mx2"),
    )
    app.dependency_overrides[get_consensus_runtime] = lambda: runtime
    response = TestClient(app).post("/api/pull", json={"model": "hf.co/x/y:z"})
    assert (
        response.status_code == 409 and "mx2 owns the model-pull lease" in response.json()["error"]
    )


def test_voters_are_changed_through_the_endpoints():
    calls: list[tuple] = []
    runtime = SimpleNamespace(
        add_voter=lambda node_id, address: calls.append(("add", node_id, address)) or True,
        remove_voter=lambda node_id, leave=False: calls.append(("remove", node_id)) or False,
    )
    app = MatricxonApp().build()
    app.dependency_overrides[get_consensus_runtime] = lambda: runtime
    client = TestClient(app)
    assert client.post(
        "/api/consensus/voters", json={"id": "n2", "address": "http://n2:8420/"}
    ).json() == {"ok": True}
    assert client.delete("/api/consensus/voters/n2").json() == {"ok": False}
    assert calls == [
        ("add", "n2", "http://n2:8420"),
        ("remove", "n2"),
    ]


def test_deleting_a_model_needs_the_pull_lease_like_pulling_one():
    from app.routers.delete_router import delete_model
    from app.schemas.delete import DeleteRequest

    other = SimpleNamespace(holds=lambda name: False)
    runtime = SimpleNamespace(keeper=other, leases=SimpleNamespace(holder=lambda name: "seed2"))
    touched: list[str] = []
    catalog = SimpleNamespace(
        delete=lambda tag: touched.append(tag), check_deletable=lambda tag: None
    )
    manager = SimpleNamespace(unload=lambda tag: touched.append(tag))
    with pytest.raises(NotLeaseHolderError):
        delete_model(DeleteRequest(model="m"), catalog, manager, runtime)
    assert touched == []
    owner = SimpleNamespace(keeper=SimpleNamespace(holds=lambda name: True), leases=None)
    delete_model(DeleteRequest(model="m"), catalog, manager, owner)
    assert touched == ["m", "m"]


def test_copying_or_retagging_a_model_needs_the_pull_lease_too():
    from app.routers.copy_router import post_copy
    from app.routers.create_router import post_create
    from app.schemas.copy import CopyRequest
    from app.schemas.create import CreateRequest

    other = SimpleNamespace(
        keeper=SimpleNamespace(holds=lambda name: False),
        leases=SimpleNamespace(holder=lambda name: "seed2"),
    )
    owner = SimpleNamespace(keeper=SimpleNamespace(holds=lambda name: True), leases=None)
    copies: list[tuple[str, str]] = []
    catalog = SimpleNamespace(copy=lambda a, b: copies.append((a, b)))
    copy_request = CopyRequest(source="a", destination="b")
    create_request = CreateRequest.model_validate({"model": "b", "from": "a"})
    with pytest.raises(NotLeaseHolderError):
        post_copy(copy_request, catalog, other)
    with pytest.raises(NotLeaseHolderError):
        post_create(create_request, catalog, other)
    assert copies == []
    post_copy(copy_request, catalog, owner)
    post_create(create_request, catalog, owner)
    assert copies == [("a", "b"), ("a", "b")]
