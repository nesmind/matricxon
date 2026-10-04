import os
import stat
import time
from pathlib import Path

import pytest
import torch

from app.runtime import cache_store, slot_codec
from app.runtime.cache_crypto import CacheCipher
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_store import PersistentCacheStore
from app.runtime.kv_cache import KVCache
from app.runtime.prompt_slot import CacheSlot

KEY = bytes(range(32))
MODEL = "m|1"
CTX = 128
DTYPE = torch.float32


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


class StubArchitecture:
    def build_cache(self, max_seq_len: int, dtype: torch.dtype) -> KVCache:
        return KVCache([(2, 4)] * 3, max_seq_len, dtype)


def _slot(token_ids: list[int]) -> CacheSlot:
    cache = KVCache([(2, 4)] * 3, CTX, DTYPE)
    cache.update(0, torch.randn(1, 2, len(token_ids), 4), torch.randn(1, 2, len(token_ids), 4))
    for layer in (1, 2):
        cache.update(
            layer, torch.randn(1, 2, len(token_ids), 4), torch.randn(1, 2, len(token_ids), 4)
        )
    cache.advance(len(token_ids))
    return CacheSlot(cache, (CTX, DTYPE), list(token_ids), prompt_len=len(token_ids))


def _store(tmp_path: Path, **policy: int | bool) -> PersistentCacheStore:
    settings = {"enabled": True, "budget_mb": 100, "ttl_hours": 24} | policy
    return PersistentCacheStore(tmp_path / "c", CacheCipher(KEY), PersistencePolicy(**settings))


def _spill(store: PersistentCacheStore, token_ids: list[int], model: str = MODEL) -> None:
    store.spill_async(model, _slot(token_ids))
    store.flush()


def test_spilled_slot_is_encrypted_private_and_restorable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    ids = list(range(1000, 1024))
    _spill(store, ids)

    (file,) = (tmp_path / "c").glob("*.mxc")
    assert stat.S_IMODE(file.stat().st_mode) == 0o600
    raw = file.read_bytes()
    assert b"1000" not in raw and b"m|1" not in raw  # neither tokens nor model id in the clear
    assert store.usage()[0] == 1

    payload = store.take(MODEL, ids + [5, 6], CTX, DTYPE, min_tokens=0)
    assert payload is not None
    slot = slot_codec.SlotCodec.decode(payload, StubArchitecture(), CTX, DTYPE)
    assert slot.token_ids == ids and slot.cache.length == 24
    assert store.usage() == (0, 0) and not list((tmp_path / "c").glob("*.mxc"))


def test_take_needs_a_real_prefix_match_and_leaves_the_file_otherwise(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    assert store.take(MODEL, [9] + list(range(1, 30)), CTX, DTYPE, 0) is None  # first block differs
    assert store.take("other|1", list(range(30)), CTX, DTYPE, 0) is None  # other model
    assert store.take(MODEL, list(range(30)), 64, DTYPE, 0) is None  # other num_ctx
    assert store.take(MODEL, list(range(30)), CTX, DTYPE, min_tokens=24) is None  # RAM has more
    assert store.usage()[0] == 1
    assert store.take(MODEL, list(range(30)), CTX, DTYPE, 0) is not None


def test_longest_shared_prefix_wins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(16)) + [700 + i for i in range(8)])
    _spill(store, list(range(24)))
    payload = store.take(MODEL, list(range(40)), CTX, DTYPE, 0)
    slot = slot_codec.SlotCodec.decode(payload, StubArchitecture(), CTX, DTYPE)
    assert slot.token_ids == list(range(24))


def test_disabled_policy_neither_writes_nor_serves(tmp_path: Path) -> None:
    store = _store(tmp_path, enabled=False)
    _spill(store, list(range(24)))
    assert store.usage()[0] == 0
    store.policy.enabled = True
    _spill(store, list(range(24)))
    store.policy.enabled = False
    assert store.take(MODEL, list(range(30)), CTX, DTYPE, 0) is None


def test_short_slots_and_oversized_files_are_not_stored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(4)))  # under MIN_PERSIST_TOKENS
    store.policy.budget_mb = 0
    _spill(store, list(range(24)))
    assert store.usage()[0] == 0


class ByteBudgetPolicy(PersistencePolicy):
    """A budget finer than the real policy's whole megabytes."""

    byte_budget: int = 10**9

    @property
    def budget_bytes(self) -> int:
        return self.byte_budget


def test_budget_drops_the_oldest_files_first(tmp_path: Path) -> None:
    policy = ByteBudgetPolicy(True, 100, 24)
    store = PersistentCacheStore(tmp_path / "c", CacheCipher(KEY), policy)
    for start in (100, 200, 300):
        _spill(store, list(range(start, start + 24)))
    ages = {}
    for age, name in enumerate(sorted(store._index, key=lambda n: store._index[n].saved_at)):
        store._index[name].saved_at = time.time() - 100 + age  # strictly increasing
        ages[age] = name
    newest_two = sum(store._index[ages[i]].size for i in (1, 2))
    policy.byte_budget = newest_two  # room for exactly the two newest files
    store.enforce_limits()
    assert store.usage()[0] == 2
    assert ages[0] not in store._index and ages[1] in store._index and ages[2] in store._index


def test_files_past_the_ttl_are_dropped(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    (name,) = store._index
    store._index[name].saved_at = time.time() - 25 * 3600
    store.enforce_limits()
    assert store.usage()[0] == 0


def test_a_restart_rebuilds_the_index_but_a_new_key_drops_everything(tmp_path: Path) -> None:
    _spill(_store(tmp_path), list(range(24)))
    assert _store(tmp_path).usage()[0] == 1
    other_key = PersistentCacheStore(
        tmp_path / "c", CacheCipher(bytes(32)), PersistencePolicy(True, 100, 24)
    )
    assert other_key.usage()[0] == 0 and not list((tmp_path / "c").glob("*.mxc"))


def test_a_tampered_body_is_rejected_and_deleted(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    (file,) = (tmp_path / "c").glob("*.mxc")
    blob = bytearray(file.read_bytes())
    blob[-5] ^= 0xFF
    file.write_bytes(bytes(blob))
    assert store.take(MODEL, list(range(30)), CTX, DTYPE, 0) is None
    assert not file.exists()


def test_clear_and_stray_tmp_cleanup(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    (tmp_path / "c" / "x.tmp").write_bytes(b"junk")
    store.clear()
    assert store.usage()[0] == 0
    assert _store(tmp_path) and not os.listdir(tmp_path / "c")


def test_forget_drops_only_that_chats_files_and_refuses_its_later_spills(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mine, other = _slot(list(range(100, 124))), _slot(list(range(200, 224)))
    mine.tag, other.tag = "chat-a", "chat-b"
    for slot in (mine, other):
        store.spill_async(MODEL, slot)
    store.flush()
    assert store.usage()[0] == 2

    assert store.forget("chat-a") == 1
    assert store.usage()[0] == 1
    assert store.take(MODEL, list(range(100, 124)) + [1], CTX, DTYPE, 0) is None

    store.spill_async(MODEL, _slot_with_tag(list(range(100, 124)), "chat-a"))
    store.flush()
    assert store.usage()[0] == 1  # the deleted chat's late spill is dropped
    assert store.forget("") == 0


def _slot_with_tag(token_ids: list[int], tag: str) -> CacheSlot:
    slot = _slot(token_ids)
    slot.tag = tag
    return slot


def test_tag_survives_a_restart_scan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.spill_async(MODEL, _slot_with_tag(list(range(24)), "chat-a"))
    store.flush()

    reopened = _store(tmp_path)
    assert reopened.forget("chat-a") == 1
