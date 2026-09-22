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
        manager.get_or_load("model-a:latest")  # refreshes before the 300s default expires

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

        manager.get_or_load("model-c:latest")  # over max_loaded=2 - evicts model-a (LRU)

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-b:latest", "model-c:latest"}

    def test_touching_a_model_protects_it_from_lru_eviction(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        manager.get_or_load("model-a:latest")
        clock.advance(1)
        manager.get_or_load("model-b:latest")
        clock.advance(1)
        manager.get_or_load("model-a:latest")  # now most-recently-used, not model-b

        manager.get_or_load("model-c:latest")

        tags = {handle.tag for handle in manager.list_loaded()}
        assert tags == {"model-a:latest", "model-c:latest"}


class TestModelManagerBusyEviction:
    """Regression guard for the real bug this was written to fix: a keep-alive-expired or
    LRU-evicted handle whose worker was still actively generating used to be silently dropped from
    the manager's own dict anyway, orphaning that worker thread (still burning CPU, invisible to
    /api/ps, unreachable by unload()'s own stop signal) exactly as if nothing were running there at
    all. Marks a handle busy the same way a real in-flight generation would - blocking its worker's
    generator mid-run via a threading.Event handshake - so this is a deterministic proof, not a
    timing-dependent guess."""

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
            clock.advance(11)  # well past keep_alive_seconds=10

            tags = {h.tag for h in manager.list_loaded()}  # calls _evict_expired internally
            assert "model-a:latest" in tags
        finally:
            release.set()

    @pytest.mark.asyncio
    async def test_evict_least_recently_used_skips_a_busy_handle(
        self, manager: ModelManager, clock: FakeClock
    ) -> None:
        handle_a = manager.get_or_load("model-a:latest")  # the LRU candidate once b/c load
        started, release = self._mark_busy(handle_a)
        try:
            await asyncio.to_thread(started.wait)
            clock.advance(1)
            manager.get_or_load("model-b:latest")
            clock.advance(1)
            manager.get_or_load("model-c:latest")  # over max_loaded=2 - model-a is LRU but busy

            tags = {h.tag for h in manager.list_loaded()}
            assert "model-a:latest" in tags  # protected despite being the LRU candidate
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

            manager.get_or_load("model-c:latest")  # nothing safe to evict - over max_loaded now

            tags = {h.tag for h in manager.list_loaded()}
            assert tags == {"model-a:latest", "model-b:latest", "model-c:latest"}
        finally:
            release_a.set()
            release_b.set()


class TestModelManagerCancelledLoad:
    """Regression guard for the deeper version of the same bug TestModelManagerUnload covers: a
    stop request that arrives *before* a handle even exists yet (i.e. while get_or_load() is still
    stuck inside the - on this project's target hardware, routinely tens-of-seconds - loading step)
    used to be silently lost entirely, since unload() had nothing in `_handles` to signal. See
    ModelManager.pop_cancelled_load/unload's own docstrings."""

    def test_unload_with_no_handle_marks_the_tag_cancelled(self, manager: ModelManager) -> None:
        manager.unload("model-a:latest")  # no handle exists yet
        assert manager.pop_cancelled_load("model-a:latest") is True

    def test_pop_cancelled_load_consumes_the_flag(self, manager: ModelManager) -> None:
        manager.unload("model-a:latest")
        assert manager.pop_cancelled_load("model-a:latest") is True
        assert manager.pop_cancelled_load("model-a:latest") is False  # already consumed

    def test_pop_cancelled_load_is_false_when_nothing_was_cancelled(
        self, manager: ModelManager
    ) -> None:
        assert manager.pop_cancelled_load("model-a:latest") is False

    def test_get_or_load_clears_a_stale_cancellation_from_an_earlier_attempt(
        self, manager: ModelManager
    ) -> None:
        manager.unload("model-a:latest")  # marks cancelled (no handle yet)
        manager.get_or_load("model-a:latest")  # a fresh, unrelated later attempt for the same tag

        assert manager.pop_cancelled_load("model-a:latest") is False

    @pytest.mark.asyncio
    async def test_unload_during_an_in_progress_load_is_noticed_once_loading_finishes(
        self, manager: ModelManager, monkeypatch
    ) -> None:
        """The actual real-world race: unload() is called on a different thread/request while
        get_or_load() is synchronously blocked inside _load() for the same tag - proven here by
        blocking the real _load() with a threading.Event handshake instead of guessing at timing."""
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
            # get_or_load() is stuck inside _load() right now - no handle exists in the manager yet.
            assert manager.list_loaded() == []

            manager.unload("model-a:latest")  # the real bug scenario
        finally:
            proceed.set()
            await asyncio.to_thread(thread.join)

        assert result["handle"].tag == "model-a:latest"  # loading still completed and is cached
        assert manager.list_loaded()[0].tag == "model-a:latest"
        assert manager.pop_cancelled_load("model-a:latest") is True


class TestModelManagerUnload:
    def test_unload_removes_a_loaded_model(self, manager: ModelManager) -> None:
        manager.get_or_load("model-a:latest")
        manager.unload("model-a:latest")

        assert manager.list_loaded() == []

    def test_unload_of_a_never_loaded_tag_is_a_no_op(self, manager: ModelManager) -> None:
        manager.unload("never-loaded:latest")  # must not raise

    def test_unload_requests_stop_on_the_handles_worker(self, manager: ModelManager) -> None:
        """Regression guard: unload() used to only pop the dict entry, leaving a mid-generation
        worker thread completely unaware anything happened - it kept computing every remaining
        token, orphaned from the manager but still pegging the CPU. See
        app.models.worker.ModelWorker.request_stop's own docstring for the full story."""
        handle = manager.get_or_load("model-a:latest")
        calls = []
        handle.worker.request_stop = lambda: calls.append(True)

        manager.unload("model-a:latest")

        assert calls == [True]
