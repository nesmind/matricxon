"""One decode step for several in-flight replies.

`ChatEngine.stream_steps` yields a `DecodeStep` when a reply needs the logits for its newest token;
the worker collects every active reply's pending step and gives them to `BatchDecoder.decode`, which
runs one forward pass for all when the architecture supports it
(`ModelArchitecture.SUPPORTS_BATCHED_DECODE`, `batch_cache`), so linear layers read each weight once
per batch. Otherwise, or for a lone reply, each step is an ordinary single-sequence forward, with
the same numbers.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.quantized_linear import QuantizedLinear
from app.runtime.batch_cache import batch_cache_for

logger = logging.getLogger(__name__)


@dataclass
class DecodeStep:
    """A reply asking for the logits of `token_id` at `position`, using (and extending) `cache`."""

    token_id: int
    cache: object
    position: int
    stop_check: Callable[[], bool] | None = None


class Checkpoint:
    """Yielded between prefill pieces: the reply is still running, but the scheduler may give other
    replies a turn (one long prompt must not stall everyone's decoding)."""


CHECKPOINT = Checkpoint()


class BatchDecoder:
    def __init__(self, architecture: ModelArchitecture, max_batch: int = 8) -> None:
        self._architecture = architecture
        self._max_batch = max(1, max_batch)
        self._can_batch: bool | None = None

    def can_batch(self) -> bool:
        """Batching needs an architecture that supports it and matmuls that profit from several
        tokens: packed weights on the Numba kernels would dequantize a whole matrix per call."""
        if self._can_batch is None:
            numba_packed = any(
                isinstance(m, QuantizedLinear) and m._native is None
                for m in self._architecture.modules()
            )
            self._can_batch = self._architecture.SUPPORTS_BATCHED_DECODE and not numba_packed
        return self._can_batch

    def decode(self, steps: list[DecodeStep]) -> list[torch.Tensor | Exception]:
        """Logits `(1, 1, vocab)` for each step, in order, or the exception that step raised."""
        results: list[torch.Tensor | Exception] = []
        for start in range(0, len(steps), self._max_batch):
            results.extend(self._decode_chunk(steps[start : start + self._max_batch]))
        return results

    def _decode_chunk(self, steps: list[DecodeStep]) -> list[torch.Tensor | Exception]:
        wrapper = batch_cache_for([s.cache for s in steps]) if len(steps) > 1 else None
        if wrapper is None or not self.can_batch():
            return [self.decode_alone(step) for step in steps]
        started = time.monotonic()
        tokens = torch.tensor([[s.token_id] for s in steps], dtype=torch.long)
        positions = torch.tensor([[s.position] for s in steps], dtype=torch.long)
        try:
            with torch.no_grad():
                logits = self._architecture.forward(
                    tokens, wrapper, positions, None, None, last_logits_only=True
                )
        except Exception as exc:  # noqa: BLE001 - delivered to every reply in this batch
            return [exc] * len(steps)
        logger.info(
            "batched decode step: %d sequences in %.2fs", len(steps), time.monotonic() - started
        )
        return [logits[b : b + 1] for b in range(len(steps))]

    def decode_alone(self, step: DecodeStep) -> torch.Tensor | Exception:
        """The ordinary single-sequence decode forward (also the fallback inside `decode`)."""
        started = time.monotonic()
        try:
            with torch.no_grad():
                logits = self._architecture.forward(
                    torch.tensor([[step.token_id]], dtype=torch.long),
                    step.cache,
                    torch.tensor([step.position], dtype=torch.long),
                    step.stop_check,
                    last_logits_only=True,
                )
        except GenerationCancelledError as exc:
            return exc
        except Exception as exc:  # noqa: BLE001 - this reply fails; the others in the round go on
            logger.exception("decode step failed")
            return exc
        logger.info("decode step forward: %.2fs", time.monotonic() - started)
        return logits
