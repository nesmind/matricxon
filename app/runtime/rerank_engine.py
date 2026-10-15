"""Scores how well each document answers a query with a scoring model (score_head.py). One forward
pass per document, like EmbeddingEngine: correct, not batched."""

import torch

from app.architectures.base import ModelArchitecture
from app.architectures.score_head import ScoreHead
from app.models.handle import ModelHandle, Tokenizer
from app.server.errors import PromptTooLongError, UnsupportedArchitectureError, WrongModelKindError

DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


class ModelKind:
    """Keeps chat away from scoring models and rerank away from chat models."""

    @staticmethod
    def labels(architecture: ModelArchitecture) -> list[str] | None:
        return getattr(architecture, "score_labels", None)

    @classmethod
    def require_generative(cls, handle: ModelHandle) -> None:
        if cls.labels(handle.architecture):
            raise WrongModelKindError(
                f"{handle.tag} is a reranking model; use /api/rerank instead of chat."
            )

    @classmethod
    def require_scoring(cls, handle: ModelHandle) -> list[str]:
        labels = cls.labels(handle.architecture)
        if not labels:
            raise WrongModelKindError(f"{handle.tag} is not a reranking model.")
        return labels


class Qwen3RerankPrompt:
    """Qwen3-Reranker's prompt, verbatim from its model card."""

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
        'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def build(self, query: str, document: str, instruction: str) -> str:
        body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
        return self._PREFIX + body + self._SUFFIX


class RerankEngine:
    _PROMPTS = {"qwen3": Qwen3RerankPrompt}

    def __init__(self, handle: ModelHandle, max_tokens: int) -> None:
        self._labels = ModelKind.require_scoring(handle)
        self._architecture = handle.architecture
        self._tokenizer: Tokenizer = handle.tokenizer
        self._max_tokens = max_tokens
        prompt_cls = self._PROMPTS.get(self._architecture.NAME)
        if prompt_cls is None:
            raise UnsupportedArchitectureError(
                f"Reranking isn't supported yet for {self._architecture.NAME} models."
            )
        self._prompt = prompt_cls()

    def score(self, query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> float:
        token_ids = self._tokenizer.encode(self._prompt.build(query, document, instruction))
        if len(token_ids) > self._max_tokens:
            raise PromptTooLongError(
                f"query and document have {len(token_ids)} tokens, "
                f"over the {self._max_tokens} limit"
            )
        with torch.no_grad():
            logits = self._architecture.forward(
                torch.tensor([token_ids], dtype=torch.long), last_logits_only=True
            )
        return ScoreHead.relevance(logits[0, -1], self._labels)
