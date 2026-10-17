"""A saved cache restores after the model loaded in the other precision (float32 vs bf16, picked by
free RAM): the mismatch used to force a full re-read (6+ minutes for a long chat on a slow CPU)."""

from pathlib import Path

import pytest
import torch

from app.runtime import cache_store, slot_codec
from app.runtime.kv_cache import KVCache
from tests.unit.test_cache_store import CTX, MODEL, StubArchitecture, _slot, _spill, _store


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def test_a_float32_save_restores_into_a_bf16_model(tmp_path: Path) -> None:
    store = _store(tmp_path)
    saved = _slot(list(range(24)))
    expected_keys, _ = saved.cache.export(24)
    store.spill_async(MODEL, saved)
    store.flush()

    taken = store.take(MODEL, list(range(30)), torch.bfloat16, 0)
    assert taken is not None
    slot = slot_codec.SlotCodec.decode(taken[0], StubArchitecture(), CTX, torch.bfloat16)
    keys, _ = slot.cache.export(24)
    assert keys[0].dtype == torch.bfloat16
    assert torch.allclose(keys[0].float(), expected_keys[0], atol=1e-2, rtol=1e-2)


def test_on_a_tie_the_same_precision_wins(tmp_path: Path) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))  # float32
    bf16 = _slot(list(range(24)))
    bf16.cache = KVCache([(2, 4)] * 3, CTX, torch.bfloat16)
    for layer in range(3):
        bf16.cache.update(layer, torch.randn(1, 2, 24, 4), torch.randn(1, 2, 24, 4))
    bf16.cache.advance(24)
    store.spill_async(MODEL, bf16)
    store.flush()

    taken = store.take(MODEL, list(range(30)), torch.bfloat16, 0)
    slot = slot_codec.SlotCodec.decode(taken[0], StubArchitecture(), CTX, torch.bfloat16)
    assert slot.cache.export(24)[0][0].dtype == torch.bfloat16
    assert store.usage()[0] == 1  # the float32 copy is still there


def test_a_shape_mismatch_is_still_refused() -> None:
    cache = KVCache([(2, 4)] * 3, CTX, torch.bfloat16)
    wrong = [torch.randn(1, 2, 5, 8)] * 3
    with pytest.raises(ValueError, match="shape"):
        cache.load(wrong, wrong)
