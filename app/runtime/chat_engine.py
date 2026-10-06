import logging
import time
from collections.abc import Callable, Generator, Iterator

import torch

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.runtime.batch_decode import CHECKPOINT, BatchDecoder, Checkpoint, DecodeStep
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
    #: A plain KV cache is split too, so a cancel or restart mid-prefill keeps the committed
    #: pieces (tokens are recorded only after a whole forward pass) instead of losing it all.
    PLAIN_PREFILL_CHUNK = 256

    def __init__(
        self,
        architecture: ModelArchitecture,
        eos_token_ids: set[int],
        prompt_cache: PromptCache | None = None,
        prefill_chunk: int = PREFILL_CHUNK,
        plain_prefill_chunk: int = PLAIN_PREFILL_CHUNK,
    ) -> None:
        self._architecture = architecture
        self._eos_token_ids = eos_token_ids
        self._prompt_cache = prompt_cache or PromptCache()
        self._prefill_chunk = prefill_chunk
        self._plain_prefill_chunk = plain_prefill_chunk

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

    def _prefill_cuts(
        self, kv_cache: object, reused: int, prompt_len: int, has_images: bool = False
    ) -> list[int]:
        """End positions of each prefill piece: `PLAIN_PREFILL_CHUNK` tokens for a plain KV cache,
        `PREFILL_CHUNK` for a hybrid one. A prompt with images stays one piece - its cache is
        never pooled, so there is nothing to keep, and image fusion sees the whole prompt."""
        if has_images:
            return [prompt_len]
        if not isinstance(kv_cache, NemotronHHybridCache):
            step = self._plain_prefill_chunk
            return sorted({prompt_len, *range(reused + step, prompt_len, step)})
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

        A plain driver over `stream_steps`: every decode step is run right here, alone. The
        worker's scheduler drives `stream_steps` itself instead, so it can batch the decode steps
        of several replies into one forward pass.
        """
        steps = self.stream_steps(request, stop_check)
        decoder = BatchDecoder(self._architecture)
        try:
            item = next(steps)
            while True:
                if isinstance(item, DecodeStep):
                    result = decoder.decode_alone(item)
                    item = (
                        steps.throw(result) if isinstance(result, Exception) else steps.send(result)
                    )
                    continue
                if isinstance(item, int):
                    yield item
                item = steps.send(None)
        except StopIteration:
            return
        finally:
            steps.close()

    def stream_steps(
        self, request: GenerationRequest, stop_check: Callable[[], bool] | None = None
    ) -> Generator[int | DecodeStep | Checkpoint, torch.Tensor | None, None]:
        """The generation as a coroutine. Yields a generated token id (an `int`); a `CHECKPOINT`
        between prefill pieces (the reply goes on, others may take a turn); and a `DecodeStep`
        whenever it needs the logits for its newest token - the caller `send()`s them back (a
        `(1, 1, vocab)` tensor) or `throw()`s the exception that step raised.

        `stop_check` (typically from `ModelWorker.stop_check_for`) is passed straight through to
        each prefill `forward()` call, so a stop can be noticed *between decoder layers*, not just
        between tokens - real gap this closes (2026-09-21): a stop request previously had to wait
        out an entire in-progress forward pass first, which on this project's slow hardware can be
        many seconds to minutes for one large-prompt prefill. `GenerationCancelledError` (raised
        deep inside `_forward_impl`'s layer loop when that happens) is caught here and ends this
        generator cleanly - same as reaching EOS or a between-token stop, never surfaced as a real
        error. Nothing here holds `torch.no_grad()` across a `yield`: several of these generators
        interleave on one thread, and a grad-mode context spanning a suspension would leak into
        the others (the worker thread disables gradients for good instead).
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
            tag=request.cache_tag,
        )
        if reused == 0 and prompt_len > 1:
            logger.info(
                "re-reading the entire chat (%d tokens) - no cached tokens reused. %s",
                prompt_len,
                self._prompt_cache.miss_reason,
            )
        sampler = Sampler(sampling)
        max_new_tokens = (
            sampling.num_predict if sampling.num_predict >= 0 else sampling.num_ctx - prompt_len
        )

        try:
            forward_started = time.monotonic()
            cuts = self._prefill_cuts(
                kv_cache, reused, prompt_len, request.image_embeddings is not None
            )
            start = reused
            for end in cuts:
                positions = (
                    request.position_ids[:, start:end]
                    if request.position_ids is not None
                    else torch.arange(start, end, dtype=torch.long)
                )
                with torch.no_grad():
                    logits = self._architecture.forward(
                        request.input_ids[:, start:end],
                        kv_cache,
                        positions,
                        stop_check,
                        self._images_in(request.image_embeddings, start, end),
                        last_logits_only=True,
                    )
                kv_cache.advance(end - start)
                self._prompt_cache.advanced(kv_cache, prompt_ids[start:end])
                self._prompt_cache.capture(kv_cache)
                start = end
                if end != cuts[-1]:
                    yield CHECKPOINT
            logger.info(
                "prefill forward: %d prompt tokens (%d reused from the cache) in %.2fs",
                prompt_len,
                reused,
                time.monotonic() - forward_started,
            )

            generated_ids: list[int] = []
            for _ in range(max_new_tokens):
                next_token = sampler.sample(logits[0, -1, :], generated_ids)
                generated_ids.append(next_token)
                yield next_token
                if next_token in self._eos_token_ids:
                    return
                position = kv_cache.length + request.position_delta
                logits = yield DecodeStep(next_token, kv_cache, position, stop_check)
                kv_cache.advance(1)
                self._prompt_cache.advanced(kv_cache, [next_token])
        except GenerationCancelledError as exc:
            logger.info("generation cancelled mid-forward-pass: %s", exc)
            return
        finally:
            self._prompt_cache.release(kv_cache)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        token_ids = list(self.stream(request))
        finish_reason = "stop" if token_ids and token_ids[-1] in self._eos_token_ids else "length"
        return GenerationResult(token_ids=token_ids, finish_reason=finish_reason)
