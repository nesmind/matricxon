from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class SamplingConfig:
    """Field names/defaults match Ollama's `options` object.

    `seed=None` must mean genuinely random every call, never a reused
    default generator - pAIring's contract for an omitted seed.
    `num_predict=-1` means "generate until EOS or num_ctx", also matching
    Ollama's convention.
    """

    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 2048
    num_predict: int = -1
    seed: int | None = None


@dataclass(frozen=True)
class GenerationRequest:
    input_ids: torch.Tensor  # (1, prompt_len)
    sampling: SamplingConfig
    # Real LLaVA vision support (2026-09-21, LlamaArchitecture only - see
    # ModelArchitecture.forward's own docstring): each (start, embeds) pair identifies a real
    # image's projected patch embeddings and the prompt token-position span (in `input_ids`) they
    # belong at. Only ever relevant to the prefill forward pass (see ChatEngine.stream) - the
    # image's own tokens are part of the prompt, never regenerated during decode.
    image_embeddings: list[tuple[int, torch.Tensor]] | None = None


@dataclass(frozen=True)
class GenerationResult:
    token_ids: list[int]
    finish_reason: Literal["stop", "length"]
