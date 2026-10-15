"""Scoring models (rerankers, classifiers): a GGUF whose head is `cls.output.weight` (one row per
label) instead of a vocabulary-sized `output.weight`, as llama.cpp converts them. The forward pass
returns label logits, never next-token logits, so these models can't chat (`ModelKind`)."""

import torch

from app.gguf.metadata import GGUFMetadata

TENSOR = "cls.output.weight"
_RANK_POOLING = 4  # llama.cpp's LLAMA_POOLING_TYPE_RANK


class ScoreHead:
    @staticmethod
    def labels(metadata: GGUFMetadata) -> list[str]:
        """The head's label names (`<arch>.classifier.output_labels`); empty when absent."""
        value = metadata.get(metadata.arch_key("classifier.output_labels"))
        return [str(label) for label in value] if isinstance(value, list) else []

    @staticmethod
    def is_scoring(metadata: GGUFMetadata, tensor_names: set[str]) -> bool:
        pooling = metadata.get(metadata.arch_key("pooling_type"))
        return TENSOR in tensor_names or pooling == _RANK_POOLING

    @staticmethod
    def relevance(logits: torch.Tensor, labels: list[str]) -> float:
        """One relevance score in [0, 1] from one position's label logits: P("yes") over yes/no
        (Qwen3-Reranker), else the sigmoid of a single logit (a one-output cross-encoder)."""
        values = logits.float().flatten()
        if "yes" in labels and "no" in labels:
            pair = torch.stack([values[labels.index("no")], values[labels.index("yes")]])
            return float(torch.softmax(pair, dim=0)[1])
        if values.numel() == 1:
            return float(torch.sigmoid(values[0]))
        raise ValueError(f"Can't score a head with labels {labels!r}")
