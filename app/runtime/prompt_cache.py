import torch

from app.architectures.base import ModelArchitecture
from app.runtime.kv_cache import KVCache
from app.runtime.mamba_cache import HybridSnapshot, NemotronHHybridCache


class PromptCache:
    """Keeps one model's cache between generation calls, so a chat's next turn only computes the
    tokens that are new. pAIring resends the whole conversation every turn; before this every
    turn re-ran prefill over all of it (measured 2026-09-23: turn 2 of a two-turn chat prefilled 70
    tokens in 12 s, while Ollama - which reuses its cache the same way - answered in 3.6 s).

    Tracks the exact token ids whose keys/values the cache holds (positions 0..len-1), and reuses
    the longest prefix they share with the next prompt - a token-level match, so a reply that
    re-tokenizes differently just ends the match early instead of reusing anything wrong. At least
    the prompt's last token is always recomputed: its logits start the next reply.

    A plain `KVCache` can be cut back to any prefix length. A hybrid cache (`NemotronHHybridCache`:
    Mamba-2 / Gated DeltaNet recurrent state) can't - recurrent state only exists at the current
    position - so while prefilling, `ChatEngine` calls `capture()` at a few positions and this
    keeps the newest `MAX_SNAPSHOTS` of them; the next call restores the latest snapshot at or
    before the shared prefix and recomputes from there.

    Not reused (a fresh cache is built, and nothing is kept for next time) when:
    - the prompt carries image embeddings - their placeholder token ids are identical for
      different images, so an id match wouldn't mean the same content;
    - `num_ctx` or the dtype changed since the cached call;
    - a hybrid cache has no snapshot at or before the shared prefix.

    One instance per loaded model (`ModelHandle.prompt_cache`); only ever used from that model's
    own `ModelWorker` thread, which runs one generation at a time.
    """

    #: Newest recurrent snapshots kept (~50MB each for Qwen3.5-4B/9B).
    MAX_SNAPSHOTS = 4

    def __init__(self) -> None:
        self._cache: KVCache | NemotronHHybridCache | None = None
        self._key: tuple[int, torch.dtype] | None = None
        self._token_ids: list[int] = []
        self._snapshots: dict[int, HybridSnapshot] = {}
        self.reused_tokens = 0

    def acquire(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        num_ctx: int,
        dtype: torch.dtype,
        reusable: bool,
    ) -> tuple[object, int]:
        """(cache, n): a cache already holding `prompt_ids[:n]`, so prefill starts at n."""
        key = (num_ctx, dtype)
        if reusable and self._cache is not None and self._key == key:
            n = min(self._common_prefix(prompt_ids), len(prompt_ids) - 1)
            reused = self._rewind(n)
            if reused is not None:
                del self._token_ids[reused:]
                self.reused_tokens = reused
                return self._cache, reused
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        keepable = isinstance(cache, (KVCache, NemotronHHybridCache))
        self._cache = cache if reusable and keepable else None
        self._key = key
        self._token_ids = []
        self._snapshots = {}
        self.reused_tokens = 0
        return cache, 0

    def _rewind(self, n: int) -> int | None:
        """Cuts the held cache back to at most `n` tokens; the length it ended at, or None if it
        can't be (hybrid cache with no snapshot at or before `n`)."""
        if isinstance(self._cache, KVCache):
            self._cache.truncate(n)
            return n
        usable = [pos for pos in self._snapshots if pos <= n]
        if not usable:
            return None
        best = max(usable)
        self._cache.restore(self._snapshots[best])
        self._snapshots = {p: s for p, s in self._snapshots.items() if p <= best}
        return best

    def advanced(self, token_ids: list[int]) -> None:
        """Records tokens whose keys/values were just committed (after `KVCache.advance`)."""
        if self._cache is not None:
            self._token_ids.extend(token_ids)

    def capture(self, cache: object) -> None:
        """Snapshots a hybrid `cache`'s recurrent state at its current length (a no-op for any
        other cache, or one this isn't holding), keeping only the newest `MAX_SNAPSHOTS`."""
        if cache is not self._cache or not isinstance(cache, NemotronHHybridCache):
            return
        self._snapshots[cache.length] = cache.snapshot()
        for pos in sorted(self._snapshots)[: -self.MAX_SNAPSHOTS]:
            del self._snapshots[pos]

    def _common_prefix(self, prompt_ids: list[int]) -> int:
        n = 0
        for cached, new in zip(self._token_ids, prompt_ids, strict=False):
            if cached != new:
                break
            n += 1
        return n
