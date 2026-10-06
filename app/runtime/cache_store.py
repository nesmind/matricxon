import json
import logging
import os
import struct
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from app.runtime.cache_crypto import CacheCipher, CacheDecryptError
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import EncodedSlot, SlotCodec

logger = logging.getLogger(__name__)

MAGIC = b"MXC1"
SUFFIX = ".mxc"
#: Token granularity of prefix matching between a saved cache and a new prompt.
BLOCK_TOKENS = 64


@dataclass
class _Entry:
    """What the store knows about a file without reading its body (sealed in its header)."""

    model_id: str
    num_ctx: int
    dtype_name: str
    length: int
    blocks: list[str]
    saved_at: float
    tag: str = (
        ""  # the conversation it came from (opaque), so a deleted chat's files can be dropped
    )
    size: int = 0  # the file's size - known only once written, so not part of the sealed meta

    def meta_json(self) -> bytes:
        return json.dumps({k: v for k, v in asdict(self).items() if k != "size"}).encode()


StoredEntry = _Entry


class PersistentCacheStore:
    """Encrypted, size- and age-limited prompt-cache files on disk: the tier below the RAM pool.

    A slot is written when the pool evicts it, when its model unloads and on shutdown - never
    during a reply. File layout: `MAGIC | len | sealed meta | sealed body`, both sealed with
    AES-256-GCM (the body bound to its meta), so nothing about a conversation - not its tokens, its
    model, or its length - is readable without the key. The index is rebuilt at startup from the
    meta headers; a file that fails authentication (wrong key, corruption) is deleted.

    A stored cache is consumed when restored (`take`): from then on the RAM pool owns it and writes
    it back when it is evicted. Limits come from the live `PersistencePolicy` - oldest files go
    first when over the budget, and any file older than the TTL is dropped.
    """

    def __init__(self, directory: Path, cipher: CacheCipher, policy: PersistencePolicy) -> None:
        self._dir = directory
        self._cipher = cipher
        self.policy = policy
        self._lock = threading.Lock()
        self._index: dict[str, _Entry] = {}
        self._forgotten: set[str] = (
            set()
        )  # tags of deleted chats: a late spill of theirs is dropped
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-spill")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._scan()
        self.enforce_limits()

    # ---- reading the directory ---------------------------------------------------------------

    def _scan(self) -> None:
        for path in self._dir.glob("*.tmp"):
            path.unlink(missing_ok=True)
        for path in self._dir.glob(f"*{SUFFIX}"):
            try:
                self._index[path.name] = self._read_meta(path)
            except (CacheDecryptError, OSError, ValueError, KeyError, TypeError):
                logger.warning("dropping unreadable prompt-cache file %s", path.name)
                path.unlink(missing_ok=True)

    def _read_meta(self, path: Path) -> _Entry:
        with path.open("rb") as handle:
            if handle.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a cache file")
            (meta_len,) = struct.unpack(">I", handle.read(4))
            meta = json.loads(self._cipher.open(handle.read(meta_len), MAGIC))
        return _Entry(**meta, size=path.stat().st_size)

    # ---- writing -----------------------------------------------------------------------------

    def spill_async(self, model_id: str, slot: CacheSlot) -> None:
        """Queues `slot` to be encoded and written by the background writer; the caller must not
        touch the slot afterwards (the pool has just dropped it)."""
        if self.policy.enabled and self.policy.budget_bytes > 0:
            self._writer.submit(self._spill, model_id, slot)

    def flush(self) -> None:
        """Blocks until every queued spill is on disk (shutdown)."""
        self._writer.submit(lambda: None).result()

    def _spill(self, model_id: str, slot: CacheSlot) -> None:
        try:
            encoded = SlotCodec.encode(slot)
            if encoded is not None and slot.tag not in self._forgotten:
                self._write(model_id, encoded, slot.tag)
        except Exception:  # noqa: BLE001 - a failed spill only costs a later re-prefill
            logger.exception("persisting a prompt cache failed")

    def _write(self, model_id: str, encoded: EncodedSlot, tag: str = "") -> None:
        entry = _Entry(
            model_id,
            encoded.num_ctx,
            encoded.dtype_name,
            len(encoded.token_ids),
            self._cipher.prefix_fingerprints(encoded.token_ids, BLOCK_TOKENS),
            time.time(),
            tag,
        )
        meta = self._cipher.seal(entry.meta_json(), MAGIC)
        body = self._cipher.seal(encoded.payload, MAGIC + meta)
        blob = MAGIC + struct.pack(">I", len(meta)) + meta + body
        if len(blob) > self.policy.budget_bytes:
            return
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".tmp")  # mkstemp creates it 0600
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
        name = f"{uuid.uuid4().hex}{SUFFIX}"
        os.replace(tmp, self._dir / name)
        entry.size = len(blob)
        with self._lock:
            self._index[name] = entry
            gone = tag in self._forgotten
        if gone:  # the chat was deleted while this was being written
            self._remove(name)
        self.enforce_limits()

    # ---- restoring ---------------------------------------------------------------------------

    def take(
        self,
        model_id: str,
        prompt_ids: list[int],
        dtype: torch.dtype,
        min_tokens: int,
    ) -> tuple[bytes, str] | None:
        """The serialized slot sharing the longest prefix (more than `min_tokens`) with
        `prompt_ids`, removed from disk, with the chat tag it was saved under - or None. Matching
        uses the sealed per-block hashes, so a file that would not help is never read, let alone
        consumed."""
        if not self.policy.enabled or not self._index:
            return None
        wanted = self._cipher.prefix_fingerprints(prompt_ids, BLOCK_TOKENS)
        best: tuple[int, str, str] | None = None
        with self._lock:
            for name, entry in self._index.items():
                if (entry.model_id, entry.dtype_name) != (model_id, str(dtype)):
                    continue
                shared = 0
                for mine, theirs in zip(wanted, entry.blocks, strict=False):
                    if mine != theirs:
                        break
                    shared += 1
                if shared * BLOCK_TOKENS > min_tokens and (best is None or shared > best[0]):
                    best = (shared, name, entry.tag)
        if best is None:
            return None
        body = self._consume(best[1])
        return None if body is None else (body, best[2])

    def inspect(self, prompt_ids: list[int]) -> tuple[list[StoredEntry], list[str]]:
        """The files' headers plus `prompt_ids`' block hashes, to explain a miss."""
        with self._lock:
            entries = list(self._index.values())
        return entries, self._cipher.prefix_fingerprints(prompt_ids, BLOCK_TOKENS)

    def _consume(self, name: str) -> bytes | None:
        path = self._dir / name
        try:
            with path.open("rb") as handle:
                handle.seek(len(MAGIC))
                (meta_len,) = struct.unpack(">I", handle.read(4))
                meta = handle.read(meta_len)
                body = self._cipher.open(handle.read(), MAGIC + meta)
        except (CacheDecryptError, OSError, struct.error):
            logger.warning("prompt-cache file %s unreadable, dropping it", name)
            self._remove(name)
            return None
        self._remove(name)
        return body

    # ---- limits and admin --------------------------------------------------------------------

    def _remove(self, name: str) -> None:
        with self._lock:
            self._index.pop(name, None)
        (self._dir / name).unlink(missing_ok=True)

    def enforce_limits(self) -> None:
        """Drops files past the TTL, then the oldest until the total is within the budget."""
        with self._lock:
            now = time.time()
            doomed = [
                n for n, e in self._index.items() if now - e.saved_at > self.policy.ttl_seconds
            ]
            keep = sorted(
                ((n, e) for n, e in self._index.items() if n not in doomed),
                key=lambda item: item[1].saved_at,
            )
            total = sum(e.size for _, e in keep)
            while keep and total > self.policy.budget_bytes:
                name, entry = keep.pop(0)
                doomed.append(name)
                total -= entry.size
        for name in doomed:
            self._remove(name)

    def forget(self, tag: str) -> int:
        """Deletes every stored file of the chat `tag` (and refuses its later spills); returns how
        many files went."""
        if not tag:
            return 0
        with self._lock:
            self._forgotten.add(tag)
            names = [n for n, e in self._index.items() if e.tag == tag]
        for name in names:
            self._remove(name)
        return len(names)

    def clear(self) -> None:
        with self._lock:
            names = list(self._index)
        for name in names:
            self._remove(name)

    def usage(self) -> tuple[int, int]:
        """(files, total bytes) currently stored."""
        with self._lock:
            return len(self._index), sum(e.size for e in self._index.values())
