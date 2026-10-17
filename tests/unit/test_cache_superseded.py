"""A chat's newer saved copy replaces the older ones it covers."""

from pathlib import Path

import pytest

from app.runtime import cache_store, slot_codec
from tests.unit.test_cache_store import MODEL, _slot, _store


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _spill(
    store: cache_store.PersistentCacheStore, ids: list[int], tag: str, model: str = MODEL
) -> None:
    slot = _slot(ids)
    slot.tag = tag
    store.spill_async(model, slot)
    store.flush()


def test_a_longer_copy_of_the_same_chat_replaces_the_older_one(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(16)), "chat-a")
    _spill(store, list(range(32)), "chat-a")
    assert store.usage()[0] == 1
    assert store.chat_count() == 1


def test_another_branch_of_the_chat_is_kept(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)), "chat-a")
    _spill(store, list(range(8)) + [500 + i for i in range(24)], "chat-a")  # an edited message
    assert store.usage()[0] == 2
    assert store.chat_count() == 1


def test_other_chats_and_models_are_untouched(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(16)), "chat-a")
    _spill(store, list(range(16)), "chat-b")
    _spill(store, list(range(16)), "chat-a", model="other|1")
    _spill(store, list(range(32)), "chat-a")
    assert store.usage()[0] == 3
    assert store.chat_count() == 2


def test_untagged_files_are_never_removed_as_superseded(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(16)), "")
    _spill(store, list(range(32)), "")
    assert store.usage()[0] == 2
    assert store.chat_count() == 2
