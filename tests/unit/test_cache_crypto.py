import json
import stat
from pathlib import Path

import pytest

from app.runtime.cache_crypto import CacheCipher, CacheDecryptError, CacheKeyStore
from app.runtime.cache_policy import PersistencePolicy

KEY = bytes(range(32))


def test_seal_open_round_trip_and_nonce_is_fresh() -> None:
    cipher = CacheCipher(KEY)
    first, second = cipher.seal(b"secret", b"aad"), cipher.seal(b"secret", b"aad")
    assert first != second
    assert b"secret" not in first
    assert cipher.open(first, b"aad") == b"secret"


def test_wrong_key_tampering_and_wrong_aad_are_rejected() -> None:
    sealed = CacheCipher(KEY).seal(b"secret", b"aad")
    with pytest.raises(CacheDecryptError):
        CacheCipher(bytes(32)).open(sealed, b"aad")
    with pytest.raises(CacheDecryptError):
        CacheCipher(KEY).open(sealed[:-1] + bytes([sealed[-1] ^ 1]), b"aad")
    with pytest.raises(CacheDecryptError):
        CacheCipher(KEY).open(sealed, b"other")


def test_key_must_be_32_bytes() -> None:
    with pytest.raises(ValueError):
        CacheCipher(b"short")


def test_prefix_fingerprints_match_exactly_on_shared_whole_blocks() -> None:
    cipher = CacheCipher(KEY)
    a = cipher.prefix_fingerprints(list(range(20)), 8)
    b = cipher.prefix_fingerprints(list(range(16)) + [999, 998, 997, 996], 8)
    assert len(a) == 2 and len(b) == 2  # the trailing partial block is not hashed
    assert a == b
    c = cipher.prefix_fingerprints([5] + list(range(1, 20)), 8)
    assert c[0] != a[0] and c[1] != a[1]  # a different first block changes every later hash
    assert CacheCipher(bytes(32)).prefix_fingerprints(list(range(20)), 8) != a


def test_key_store_generates_a_private_key_file_once(tmp_path: Path) -> None:
    key_file = tmp_path / "sub" / "k.key"
    sealed = CacheKeyStore(key_file).load().seal(b"x")
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
    assert CacheKeyStore(key_file).load().open(sealed) == b"x"  # same key on the next start


def test_key_store_env_passphrase_wins(tmp_path: Path) -> None:
    sealed = CacheKeyStore(tmp_path / "k.key", "pass").load().seal(b"x")
    assert not (tmp_path / "k.key").exists()
    assert CacheKeyStore(tmp_path / "other.key", "pass").load().open(sealed) == b"x"


def test_policy_defaults_then_saved_values_win(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    policy = PersistencePolicy.load(path, True, 100, 24)
    assert (policy.enabled, policy.budget_mb, policy.ttl_hours) == (True, 100, 24)
    policy.update(enabled=False, budget_mb=7)
    reloaded = PersistencePolicy.load(path, True, 100, 24)
    assert (reloaded.enabled, reloaded.budget_mb, reloaded.ttl_hours) == (False, 7, 24)


def test_policy_ignores_a_corrupt_file_and_clamps(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text("{nope")
    assert PersistencePolicy.load(path, True, 5, 3).budget_mb == 5
    path.write_text(json.dumps({"enabled": True, "budget_mb": -4, "ttl_hours": 0}))
    policy = PersistencePolicy.load(path, False, 5, 3)
    assert (policy.budget_mb, policy.ttl_hours) == (0, 1)
