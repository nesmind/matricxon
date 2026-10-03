"""Encryption of persisted prompt caches: AES-256-GCM, so a copied or backed-up cache directory is
unreadable (and tamper-evident) without the key."""

import hashlib
import hmac
import os
from array import array
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_BYTES = 32
NONCE_BYTES = 12


class CacheDecryptError(Exception):
    """Wrong key, corrupted file, or a file that was modified."""


class CacheCipher:
    def __init__(self, key: bytes) -> None:
        if len(key) != KEY_BYTES:
            raise ValueError(f"cache key must be {KEY_BYTES} bytes")
        self._aead = AESGCM(key)
        # A separate sub-key for fingerprints, so one key never serves two purposes.
        self._mac_key = hashlib.sha256(b"matricxon-cache-mac" + key).digest()

    def seal(self, plaintext: bytes, aad: bytes = b"") -> bytes:
        """`nonce || ciphertext+tag`; a fresh random nonce every call."""
        nonce = os.urandom(NONCE_BYTES)
        return nonce + self._aead.encrypt(nonce, plaintext, aad)

    def open(self, sealed: bytes, aad: bytes = b"") -> bytes:
        try:
            return self._aead.decrypt(sealed[:NONCE_BYTES], sealed[NONCE_BYTES:], aad)
        except InvalidTag as exc:
            raise CacheDecryptError("cache file failed authentication") from exc

    def prefix_fingerprints(self, token_ids: list[int], block: int) -> list[str]:
        """One short keyed hash per whole `block` of tokens: entry i covers `token_ids[:(i+1) *
        block]`. Two token lists share their first k blocks exactly when the first k entries are
        equal - how a saved cache is matched to a new prompt without being decrypted first."""
        running = hmac.new(self._mac_key, digestmod=hashlib.sha256)
        hashes: list[str] = []
        for start in range(0, len(token_ids) - block + 1, block):
            running.update(array("i", token_ids[start : start + block]).tobytes())
            hashes.append(running.copy().hexdigest()[:16])
        return hashes


class CacheKeyStore:
    """Where the key comes from: `MATRICXON_CACHE_KEY` if set (any passphrase, hashed to 32
    bytes), else a random key generated once into a 0600 file."""

    def __init__(self, key_file: Path, env_value: str | None = None) -> None:
        self._key_file = key_file
        self._env_value = env_value

    def load(self) -> CacheCipher:
        if self._env_value:
            return CacheCipher(hashlib.sha256(self._env_value.encode()).digest())
        if self._key_file.exists():
            return CacheCipher(self._key_file.read_bytes())
        self._key_file.parent.mkdir(parents=True, exist_ok=True)
        key = os.urandom(KEY_BYTES)
        fd = os.open(self._key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(key)
        return CacheCipher(key)
