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
    """Runs one autoregressive generation call: prefill the prompt, then sample and decode one token
    at a time.

    Lifetime is one call, but the KV cache can outlive it: given the model's `PromptCache`, prefill
    computes only the tokens past the prefix the previous call cached. Without one every call builds
    a fresh cache. The Sampler is always fresh. `architecture` must be a decoder built via
    `build_cache` (`ModelArchitecture.build_cache`, a hook because only `nemotron_h`'s hybrid layers
    need a non-plain cache); encoder-only architectures use EmbeddingEngine.
    """

    # A hybrid (recurrent) cache's prefill is split into pieces of at most this many tokens, with a
    # snapshot after each (`PromptCache`); this also bounds peak activation memory on a long prompt.
    # Plus one extra cut this many tokens before the prompt's end: the next turn's prompt usually
    # diverges just before the generation-prompt suffix, the useful restore point.
    PREFILL_CHUNK = 512
    TAIL_SNAPSHOT_OFFSET = 16
    # A plain KV cache is split too, so a cancel or restart mid-prefill keeps the committed pieces
    # (tokens are recorded only after a whole forward pass) instead of losing everything.
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
        """The image spans (and parts of spans) inside prompt tokens `[start, end)`, with starts
        relative to `start`: a chunked prefill hands each piece only its own part."""
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
        `PREFILL_CHUNK` for a hybrid one. A prompt with images stays one piece: its cache is never
        pooled, so nothing to keep, and image fusion sees the whole prompt."""
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
        """Yields one generated token id at a time, what `/api/chat`'s NDJSON streaming needs
        (`generate` collects everything, for tests and the M4 smoke test). A plain driver over
        `stream_steps`: every decode step runs here, alone. The worker's scheduler drives
        `stream_steps` itself to batch several replies' decode steps into one forward pass."""
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
        """The generation as a coroutine. Yields a generated token id (`int`); a `CHECKPOINT`
        between prefill pieces (the reply goes on, others may take a turn); and a `DecodeStep` when
        it needs the logits for its newest token: the caller `send`s them back (a `(1, 1, vocab)`
        tensor) or `throw`s the step's exception.

        `stop_check` (typically `ModelWorker.stop_check_for`) is passed to each prefill `forward`,
        so a stop is noticed between decoder layers, not just tokens (a stop used to wait out a
        whole forward pass: minutes for a large-prompt prefill on slow hardware).
        `GenerationCancelledError`, raised in `_forward_impl`'s layer loop, is caught here and ends
        the generator cleanly, like EOS or a between-token stop. Nothing holds `torch.no_grad`
        across a `yield`: several generators interleave on one thread, and a grad-mode context
        spanning a suspension would leak into the others (the worker thread disables gradients for
        good).
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
        # Said before the prefill starts, so a long (or cancelled) prefill is explained too.
        if reused > 0:
            logger.info(
                "prompt cache: reusing %d of %d tokens, prefilling the other %d",
                reused,
                prompt_len,
                prompt_len - reused,
            )
        elif prompt_len > 1:
            logger.info(
                "re-reading the entire chat (%d tokens) - no cached tokens reused. %s",
                prompt_len,
                self._prompt_cache.miss_reason,
            )
        sampler = Sampler(sampling)
        room = max(sampling.num_ctx - prompt_len, 0)
        max_new_tokens = room if sampling.num_predict < 0 else min(sampling.num_predict, room)

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
