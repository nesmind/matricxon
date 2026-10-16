import asyncio
import threading
from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager
from tests.tiny_gguf import build_tiny_mistral3_gguf


class FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self._now = now

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def _install_tiny_model(models_dir: Path, tag: str) -> None:
    import json

    repo_dir = models_dir / tag.replace(":", "_")
    repo_dir.mkdir(parents=True)
    gguf_path = repo_dir / "model.gguf"
    build_tiny_mistral3_gguf(gguf_path)

    installed = InstalledModel(
        tag=tag,
        path=str(gguf_path),
        architecture="mistral3",
        capabilities=["completion"],
        size_bytes=gguf_path.stat().st_size,
        family="mistral3",
        parameter_size="0.001B",
        context_length=32,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def manager(tmp_path: Path, clock: FakeClock) -> ModelManager:
    models_dir = tmp_path / "models"
    _install_tiny_model(models_dir, "model-a:latest")
    _install_tiny_model(models_dir, "model-b:latest")
    _install_tiny_model(models_dir, "model-c:latest")
    return ModelManager(
        ModelCatalog(models_dir),
        max_loaded=2,
        default_keep_alive_seconds=300,
        clock=clock,
    )


class TestModelManagerLoading:
    def test_loading_the_same_tag_twice_returns_the_same_handle(
        self, manager: ModelManager
    ) -> None:
        first = manager.get_or_load("model-a:latest")
        second = manager.get_or_load("model-a:latest")
        assert first is second

    def test_unknown_tag_raises(self, manager: ModelManager) -> None:
        from app.server.errors import UnknownModelError

        with pytest.raises(UnknownModelError):
            manager.get_or_load("nonexistent:latest")


class TestModelManagerKeepAlive:
    def test_touch_refreshes_last_used_at(self, manager: ModelManager, clock: FakeClock) -> None:
        handle = manager.get_or_load("model-a:latest")
        clock.advance(299)
        manager.get_or_load("model-a:latest")

        assert not handle.is_expired(clock())

    def test_expires_after_default_keep_alive(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("model-a:latest")
        clock.advance(301)

        assert manager.list_loaded() == []

    def test_explicit_keep_alive_overrides_default(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("model-a:latest", keep_alive_seconds=1)
        clock.advance(2)

        assert manager.list_loaded() == []


class TestModelManagerEviction:
    def test_evicts_least_recently_used_at_capacity(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("model-a:latest")
        clock.advance(1)
        manager.get_or_load("model-b:latest")

        manager.get_or_load("model-c:latest")

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-b:latest", "model-c:latest"}

    def test_touching_a_model_protects_it_from_lru_eviction(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("model-a:latest")
        clock.advance(1)
        manager.get_or_load("model-b:latest")
        clock.advance(1)
        manager.get_or_load("model-a:latest")

        manager.get_or_load("model-c:latest")

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-a:latest", "model-c:latest"}


class TestModelManagerBusyEviction:
    """Guards the fixed bug: a keep-alive-expired or LRU-evicted handle whose worker was still
    generating was dropped from the manager's dict anyway, orphaning the worker thread (burning CPU,
    invisible to /api/ps, unreachable by unload's stop). A handle is marked busy as a real
    generation would be, blocking its worker's generator via a threading.Event handshake: a
    deterministic proof, not a timing guess."""

    @staticmethod
    def _mark_busy(handle) -> tuple[threading.Event, threading.Event]:
        started = threading.Event()
        release = threading.Event()

        def blocking_gen():
            started.set()
            release.wait()
            yield "done"

        handle.worker.stream(blocking_gen)
        return started, release

    @pytest.mark.asyncio
    async def test_evict_expired_skips_a_busy_handle(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        handle = manager.get_or_load("model-a:latest", keep_alive_seconds=10)
        started, release = self._mark_busy(handle)
        try:
            await asyncio.to_thread(started.wait)
            clock.advance(11)

            tags = {h.tag for h in manager.list_loaded()}
            assert "model-a:latest" in tags
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_keep_alive_counts_from_the_end_of_a_reply(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        """A reply longer than keep-alive used to unload the model the moment it ended."""
        handle = manager.get_or_load("model-a:latest", keep_alive_seconds=10)
        started, release = self._mark_busy(handle)
        await asyncio.to_thread(started.wait)
        clock.advance(11)
        release.set()
        await asyncio.to_thread(handle.worker.wait_until_idle)

        assert "model-a:latest" in {h.tag for h in manager.list_loaded()}
        clock.advance(11)
        assert "model-a:latest" not in {h.tag for h in manager.list_loaded()}

    @pytest.mark.asyncio
    async def test_evict_least_recently_used_skips_a_busy_handle(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        handle_a = manager.get_or_load("model-a:latest")
        started, release = self._mark_busy(handle_a)
        try:
            await asyncio.to_thread(started.wait)
            clock.advance(1)
            manager.get_or_load("model-b:latest")
            clock.advance(1)
            manager.get_or_load("model-c:latest")

            tags = {h.tag for h in manager.list_loaded()}
            assert "model-a:latest" in tags
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_evict_least_recently_used_is_a_no_op_when_every_handle_is_busy(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        handle_a = manager.get_or_load("model-a:latest")
        clock.advance(1)
        handle_b = manager.get_or_load("model-b:latest")
        started_a, release_a = self._mark_busy(handle_a)
        started_b, release_b = self._mark_busy(handle_b)
        try:
            await asyncio.to_thread(started_a.wait)
            await asyncio.to_thread(started_b.wait)

            manager.get_or_load("model-c:latest")

            tags = {h.tag for h in manager.list_loaded()}
            assert tags == {"model-a:latest", "model-b:latest", "model-c:latest"}
        finally:
            release_a.set()
            release_b.set()


class TestModelManagerCancelledLoad:
    """Guards the deeper version of the bug TestModelManagerUnload covers: a stop arriving *before*
    a handle exists (get_or_load still in the, on the target hardware tens-of-seconds, loading
    step) was lost, as unload had nothing in `_handles` to signal. See
    ModelManager.pop_cancelled_load/unload."""

    def test_unload_with_no_handle_marks_the_tag_cancelled(self, manager: ModelManager) -> None:
        manager.unload("model-a:latest")
        assert manager.pop_cancelled_load("model-a:latest") is True

    def test_pop_cancelled_load_consumes_the_flag(self, manager: ModelManager) -> None:
        manager.unload("model-a:latest")
        assert manager.pop_cancelled_load("model-a:latest") is True
        assert manager.pop_cancelled_load("model-a:latest") is False

    def test_pop_cancelled_load_is_false_when_nothing_was_cancelled(
        self, manager: ModelManager
    ) -> None:
        assert manager.pop_cancelled_load("model-a:latest") is False

    def test_get_or_load_clears_a_stale_cancellation_from_an_earlier_attempt(
        self, manager: ModelManager
    ) -> None:
        manager.unload("model-a:latest")
        manager.get_or_load("model-a:latest")

        assert manager.pop_cancelled_load("model-a:latest") is False

    @pytest.mark.asyncio
    async def test_unload_during_an_in_progress_load_is_noticed_once_loading_finishes(
        self, manager: ModelManager, monkeypatch
    ) -> None:
        """The real race: unload runs on another thread/request while get_or_load is blocked
        inside _load for the same tag; the real _load is blocked with a threading.Event
        handshake instead of guessing timing."""
        load_started = threading.Event()
        proceed = threading.Event()
        real_load = ModelManager._load

        def blocking_load(self, tag, keep_alive_seconds, now):
            load_started.set()
            proceed.wait()
            return real_load(self, tag, keep_alive_seconds, now)

        monkeypatch.setattr(ModelManager, "_load", blocking_load)

        result: dict = {}

        def do_load() -> None:
            result["handle"] = manager.get_or_load("model-a:latest")

        thread = threading.Thread(target=do_load)
        thread.start()
        try:
            await asyncio.to_thread(load_started.wait)
            # get_or_load is stuck inside _load now: no handle exists in the manager yet.
            assert manager.list_loaded() == []

            manager.unload("model-a:latest")
        finally:
            proceed.set()
            await asyncio.to_thread(thread.join)

        assert result["handle"].tag == "model-a:latest"
        assert manager.list_loaded()[0].tag == "model-a:latest"
        assert manager.pop_cancelled_load("model-a:latest") is True


class TestModelManagerUnload:
    def test_unload_removes_a_loaded_model(self, manager: ModelManager) -> None:
        manager.get_or_load("model-a:latest")
        manager.unload("model-a:latest")

        assert manager.list_loaded() == []

    def test_unload_of_a_never_loaded_tag_is_a_no_op(self, manager: ModelManager) -> None:
        manager.unload("never-loaded:latest")

    def test_unload_requests_stop_on_the_handles_worker(self, manager: ModelManager) -> None:
        """Guard: unload only popped the dict entry, leaving a mid-generation worker thread
        unaware; it kept computing every remaining token, orphaned but still pegging the CPU
        (app.models.worker.ModelWorker.request_stop)."""
        handle = manager.get_or_load("model-a:latest")
        calls = []
        handle.worker.request_stop = lambda: calls.append(True)

        manager.unload("model-a:latest")

        assert calls == [True]


def test_an_unloaded_model_is_freed(manager: ModelManager) -> None:
    """The worker thread never ended and held the model (decoder, keep-alive callback), so every
    unloaded model stayed in RAM."""
    import gc
    import weakref

    handle = manager.get_or_load("model-a:latest")
    handle_ref, architecture_ref = weakref.ref(handle), weakref.ref(handle.architecture)
    thread = handle.worker._thread
    del handle
    manager.unload("model-a:latest")
    thread.join(5)
    gc.collect()

    assert not thread.is_alive()
    assert handle_ref() is None and architecture_ref() is None
