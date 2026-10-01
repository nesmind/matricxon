import logging
import time
from collections.abc import Callable, Iterator

import torch

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.runtime.generation_request import GenerationRequest, GenerationResult
from app.runtime.mamba_cache import NemotronHHybridCache
from app.runtime.prompt_cache import PromptCache
from app.runtime.sampler import Sampler

logger = logging.getLogger(__name__)


class ChatEngine:
    """Runs one autoregressive generation call: prefill the prompt, then sample and decode one
    token at a time.

    Lifetime is a single call, but the KV cache can outlive it: given the model's `PromptCache`
    (see its docstring), prefill only computes the tokens past the prefix the previous call already
    has in the cache. Without one, every call builds a fresh cache. The Sampler is always built from
    scratch. `architecture` must be a decoder (built via its own
    `build_cache()` - see `ModelArchitecture.build_cache`'s own docstring for
    why this is a hook rather than a `KVCache` built directly here: every
    architecture but `nemotron_h`'s hybrid Mamba-2/attention/MLP layers gets
    the same plain per-layer K/V cache this always built) - encoder-only
    architectures go through EmbeddingEngine instead.
    """

    #: A hybrid (recurrent) cache's prefill is split into pieces of at most this many tokens, with
    #: a snapshot after each (see `PromptCache`) - also bounds peak activation memory on a long
    #: prompt. Plus one extra cut this many tokens before the prompt's end: the next turn's
    #: prompt usually diverges just before the generation-prompt suffix - the useful restore point.
    PREFILL_CHUNK = 512
    TAIL_SNAPSHOT_OFFSET = 16

    def __init__(
        self,
        architecture: ModelArchitecture,
        eos_token_ids: set[int],
        prompt_cache: PromptCache | None = None,
        prefill_chunk: int = PREFILL_CHUNK,
    ) -> None:
        self._architecture = architecture
        self._eos_token_ids = eos_token_ids
        self._prompt_cache = prompt_cache or PromptCache()
        self._prefill_chunk = prefill_chunk

    @staticmethod
    def _images_in(
        images: list[tuple[int, torch.Tensor]] | None, start: int, end: int
    ) -> list[tuple[int, torch.Tensor]] | None:
        """The image spans (and the parts of spans) inside prompt tokens `[start, end)`, with
        starts made relative to `start` - a chunked prefill hands each piece only its own part."""
        if not images:
            return images
        pieces = []
        for span_start, embeds in images:
            lo, hi = max(span_start, start), min(span_start + embeds.shape[0], end)
            if lo < hi:
                pieces.append((lo - start, embeds[lo - span_start : hi - span_start]))
        return pieces or None

    def _prefill_cuts(self, kv_cache: object, reused: int, prompt_len: int) -> list[int]:
        """End positions of each prefill piece. One piece (the whole remaining prompt) for a plain
        KV cache; split for a hybrid one (see `PREFILL_CHUNK`)."""
        if not isinstance(kv_cache, NemotronHHybridCache):
            return [prompt_len]
        cuts = {prompt_len, *range(reused + self._prefill_chunk, prompt_len, self._prefill_chunk)}
        if prompt_len - self.TAIL_SNAPSHOT_OFFSET > reused:
            cuts.add(prompt_len - self.TAIL_SNAPSHOT_OFFSET)
        return sorted(cuts)

    def stream(
        self, request: GenerationRequest, stop_check: Callable[[], bool] | None = None
    ) -> Iterator[int]:
        """Yields one generated token id at a time - what `/api/chat`'s NDJSON
        streaming actually needs. See `generate()` for the collect-everything
        convenience wrapper used by tests and the M4 manual smoke test.

        `stop_check` (typically `ModelWorker.should_stop`, see its own docstring) is passed
        straight through to each `forward()` call, so a stop can be noticed *between decoder
        layers*, not just between tokens the way `finally: yield` boundaries already allow -
        real gap this closes (2026-09-21): a stop request previously had to wait out an entire
        in-progress forward pass first, which on this project's slow hardware can be many seconds
        to minutes for one large-prompt prefill. `GenerationCancelledError` (raised deep inside
        `_forward_impl`'s layer loop when that happens) is caught here and ends this generator
        cleanly - same as reaching EOS or a between-token stop, never surfaced as a real error.
        """
        sampling = request.sampling
        model_dtype = next(self._architecture.parameters()).dtype
        prompt_ids = request.input_ids[0].tolist()
        prompt_len = len(prompt_ids)
        kv_cache, reused = self._prompt_cache.acquire(
            self._architecture,
            prompt_ids,
            sampling.num_ctx,
            model_dtype,
            reusable=request.image_embeddings is None,
        )
        sampler = Sampler(sampling)
        max_new_tokens = (
            sampling.num_predict if sampling.num_predict >= 0 else sampling.num_ctx - prompt_len
        )

        try:
            with torch.no_grad():
                forward_started = time.monotonic()
                cuts = self._prefill_cuts(kv_cache, reused, prompt_len)
                start = reused
                for end in cuts:
                    positions = (
                        request.position_ids[:, start:end]
                        if request.position_ids is not None
                        else torch.arange(start, end, dtype=torch.long)
                    )
                    logits = self._architecture.forward(
                        request.input_ids[:, start:end],
                        kv_cache,
                        positions,
                        stop_check,
                        self._images_in(request.image_embeddings, start, end),
                        last_logits_only=True,
                    )
                    kv_cache.advance(end - start)
                    self._prompt_cache.advanced(prompt_ids[start:end])
                    self._prompt_cache.capture(kv_cache)
                    start = end
                logger.info(
                    "prefill forward: %d prompt tokens (%d reused from the cache) in %.2fs",
                    prompt_len,
                    reused,
                    time.monotonic() - forward_started,
                )

                generated_ids: list[int] = []
                for step in range(max_new_tokens):
                    next_token = sampler.sample(logits[0, -1, :], generated_ids)
                    generated_ids.append(next_token)
                    yield next_token
                    if next_token in self._eos_token_ids:
                        return

                    next_input = torch.tensor([[next_token]], dtype=torch.long)
                    position_ids = torch.tensor(
                        [kv_cache.length + request.position_delta], dtype=torch.long
                    )
                    forward_started = time.monotonic()
                    logits = self._architecture.forward(
                        next_input, kv_cache, position_ids, stop_check, last_logits_only=True
                    )
                    logger.info(
                        "decode step %d forward: %.2fs",
                        step + 1,
                        time.monotonic() - forward_started,
                    )
                    kv_cache.advance(1)
                    self._prompt_cache.advanced([next_token])
        except GenerationCancelledError as exc:
            logger.info("generation cancelled mid-forward-pass: %s", exc)
            return

    def generate(self, request: GenerationRequest) -> GenerationResult:
        token_ids = list(self.stream(request))
        finish_reason = "stop" if token_ids and token_ids[-1] in self._eos_token_ids else "length"
        return GenerationResult(token_ids=token_ids, finish_reason=finish_reason)
