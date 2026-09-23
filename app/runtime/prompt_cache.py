import torch

from app.architectures.base import ModelArchitecture
from app.runtime.kv_cache import KVCache


class PromptCache:
    """Keeps one model's KV cache between generation calls, so a chat's next turn only computes
    the tokens that are new. pAIring resends the whole conversation every turn; before this every
    turn re-ran prefill over all of it (measured 2026-09-23: turn 2 of a two-turn chat prefilled 70
    tokens in 12 s, while Ollama - which reuses its cache the same way - answered in 3.6 s).

    Tracks the exact token ids whose keys/values the cache holds (positions 0..len-1), and reuses
    the longest prefix they share with the next prompt - a token-level match, so a reply that
    re-tokenizes differently just ends the match early instead of reusing anything wrong. At least
    the prompt's last token is always recomputed: its logits start the next reply.

    Not reused (a fresh cache is built, and nothing is kept for next time) when:
    - the prompt carries image embeddings - their placeholder token ids are identical for
      different images, so an id match wouldn't mean the same content;
    - the architecture's cache isn't a plain `KVCache` (nemotron_h's hybrid cache holds Mamba-2
      recurrent state, which can't be rolled back to a shorter prefix);
    - `num_ctx` or the dtype changed since the cached call.

    One instance per loaded model (`ModelHandle.prompt_cache`); only ever used from that model's
    own `ModelWorker` thread, which runs one generation at a time.
    """

    def __init__(self) -> None:
        self._cache: KVCache | None = None
        self._key: tuple[int, torch.dtype] | None = None
        self._token_ids: list[int] = []
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
            self._cache.truncate(n)
            del self._token_ids[n:]
            self.reused_tokens = n
            return self._cache, n
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        self._cache = cache if reusable and isinstance(cache, KVCache) else None
        self._key = key
        self._token_ids = []
        self.reused_tokens = 0
        return cache, 0

    def advanced(self, token_ids: list[int]) -> None:
        """Records tokens whose keys/values were just committed (after `KVCache.advance`)."""
        if self._cache is not None:
            self._token_ids.extend(token_ids)

    def _common_prefix(self, prompt_ids: list[int]) -> int:
        n = 0
        for cached, new in zip(self._token_ids, prompt_ids, strict=False):
            if cached != new:
                break
            n += 1
        return n
