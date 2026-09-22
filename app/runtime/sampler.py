import logging
import time

import torch

from app.runtime.generation_request import SamplingConfig

logger = logging.getLogger(__name__)


class RepetitionPenaltyFilter:
    """Divides (or, for negative logits, multiplies) already-generated
    tokens' scores by `penalty`, per Ollama/llama.cpp's `repeat_penalty`
    convention - suppresses repeats without ever fully forbidding a token.
    """

    def __init__(self, penalty: float) -> None:
        self._penalty = penalty

    def apply(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        if self._penalty == 1.0 or not generated_ids:
            return logits
        logits = logits.clone()
        indices = torch.tensor(sorted(set(generated_ids)), dtype=torch.long)
        scores = logits[indices]
        logits[indices] = torch.where(scores > 0, scores / self._penalty, scores * self._penalty)
        return logits


class TemperatureScaler:
    """`temperature <= 0` is a short-circuit to greedy argmax, matching
    Ollama's convention - the rest of the pipeline never runs in that case.
    """

    def __init__(self, temperature: float) -> None:
        self._temperature = temperature

    @property
    def is_greedy(self) -> bool:
        return self._temperature <= 0.0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self._temperature


class TopKFilter:
    def __init__(self, k: int) -> None:
        self._k = k

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._k <= 0 or self._k >= logits.shape[-1]:
            return logits
        threshold = torch.topk(logits, self._k).values[-1]
        return logits.masked_fill(logits < threshold, float("-inf"))


class TopPFilter:
    """Nucleus sampling: keeps the smallest prefix of sorted probabilities
    whose cumulative mass reaches `p`, always keeping at least the top token.
    """

    def __init__(self, p: float) -> None:
        self._p = p

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > self._p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        result = torch.full_like(logits, float("-inf"))
        result.scatter_(-1, sorted_indices, sorted_logits)
        return result


class Sampler:
    """Ollama-option-shaped sampling pipeline: repeat-penalty -> temperature
    -> top-k -> top-p -> multinomial, run once per generated token against
    that step's vocab-sized logit vector.
    """

    def __init__(self, sampling: SamplingConfig) -> None:
        self._repetition_penalty = RepetitionPenaltyFilter(sampling.repeat_penalty)
        self._temperature = TemperatureScaler(sampling.temperature)
        self._top_k = TopKFilter(sampling.top_k)
        self._top_p = TopPFilter(sampling.top_p)

        self._generator = torch.Generator()
        if sampling.seed is not None:
            self._generator.manual_seed(sampling.seed)
        else:
            self._generator.seed()

    def sample(self, logits: torch.Tensor, generated_ids: list[int]) -> int:
        stage_started = time.monotonic()
        logits = logits.to(torch.float32)
        logits = self._repetition_penalty.apply(logits, generated_ids)

        if self._temperature.is_greedy:
            token_id = int(torch.argmax(logits).item())
            logger.debug(
                "picking next word: greedy argmax -> token %d in %.2fms",
                token_id,
                (time.monotonic() - stage_started) * 1000,
            )
            return token_id

        logits = self._temperature.apply(logits)
        logits = self._top_k.apply(logits)
        logits = self._top_p.apply(logits)

        probabilities = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=self._generator)
        token_id = int(sampled.item())
        logger.debug(
            "picking next word: temperature/top-k/top-p sample -> token %d in %.2fms",
            token_id,
            (time.monotonic() - stage_started) * 1000,
        )
        return token_id
