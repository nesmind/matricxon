import logging
import time
from collections.abc import Callable, Iterator

import torch

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.runtime.generation_request import GenerationRequest, GenerationResult
from app.runtime.kv_cache import KVCache
from app.runtime.sampler import Sampler

logger = logging.getLogger(__name__)


class ChatEngine:
    """Runs one autoregressive generation call: prefill the prompt against a
    fresh KVCache, then sample and decode one token at a time.

    Lifetime is a single call - matricxon has no cross-request prompt
    caching (see ROADMAP.md), so every call builds its own cache and
    Sampler from scratch. `architecture` must be a decoder (expose
    `kv_cache_layer_shapes` - a `(n_head_kv, head_dim)` pair per layer, see
    `KVCache`'s own docstring for why it's per-layer rather than one shared
    pair - as `Mistral3TextArchitecture` does) - encoder-only architectures
    go through EmbeddingEngine instead.
    """

    def __init__(self, architecture: ModelArchitecture, eos_token_ids: set[int]) -> None:
        self._architecture = architecture
        self._eos_token_ids = eos_token_ids

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
        kv_cache = KVCache(
            layer_shapes=self._architecture.kv_cache_layer_shapes,
            max_seq_len=sampling.num_ctx,
            dtype=model_dtype,
        )
        sampler = Sampler(sampling)

        input_ids = request.input_ids
        prompt_len = input_ids.shape[1]
        max_new_tokens = (
            sampling.num_predict if sampling.num_predict >= 0 else sampling.num_ctx - prompt_len
        )

        try:
            with torch.no_grad():
                position_ids = torch.arange(prompt_len, dtype=torch.long)
                forward_started = time.monotonic()
                logits = self._architecture.forward(
                    input_ids, kv_cache, position_ids, stop_check, request.image_embeddings
                )
                logger.info(
                    "prefill forward: %d prompt tokens in %.2fs",
                    prompt_len,
                    time.monotonic() - forward_started,
                )
                kv_cache.advance(prompt_len)

                generated_ids: list[int] = []
                for step in range(max_new_tokens):
                    next_token = sampler.sample(logits[0, -1, :], generated_ids)
                    generated_ids.append(next_token)
                    yield next_token
                    if next_token in self._eos_token_ids:
                        return

                    next_input = torch.tensor([[next_token]], dtype=torch.long)
                    position_ids = torch.tensor([kv_cache.length], dtype=torch.long)
                    forward_started = time.monotonic()
                    logits = self._architecture.forward(
                        next_input, kv_cache, position_ids, stop_check
                    )
                    logger.info(
                        "decode step %d forward: %.2fs",
                        step + 1,
                        time.monotonic() - forward_started,
                    )
                    kv_cache.advance(1)
        except GenerationCancelledError as exc:
            logger.info("generation cancelled mid-forward-pass: %s", exc)
            return

    def generate(self, request: GenerationRequest) -> GenerationResult:
        token_ids = list(self.stream(request))
        finish_reason = "stop" if token_ids and token_ids[-1] in self._eos_token_ids else "length"
        return GenerationResult(token_ids=token_ids, finish_reason=finish_reason)
