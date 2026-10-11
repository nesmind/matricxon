import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import main
from app.main import MatricxonApp


def _run_lifespan(app) -> None:
    async def go() -> None:
        async with app.router.lifespan_context(app):
            pass

    asyncio.run(go())


def test_startup_builds_the_cache_store_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    monkeypatch.setattr(main, "get_cache_store", lambda: events.append("store"))
    monkeypatch.setattr(
        main, "get_model_manager", lambda: SimpleNamespace(shutdown=lambda: events.append("down"))
    )
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    _run_lifespan(MatricxonApp().build())
    assert events == ["store", "down"]


def test_a_cluster_node_without_a_key_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from app.runtime.cache_crypto import CacheKeyMissingError

    def no_key() -> None:
        raise CacheKeyMissingError("no key")

    monkeypatch.setattr(main, "get_cache_store", no_key)
    monkeypatch.setenv("MATRICXON_PID_FILE", str(tmp_path / "pid"))
    with pytest.raises(CacheKeyMissingError):
        _run_lifespan(MatricxonApp().build())
    assert not (tmp_path / "pid").exists()
