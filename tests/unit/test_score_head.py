"""score_head.py: reranker detection and the relevance formula (Qwen3-Reranker's P("yes"))."""

import math

import pytest
import torch

from app.architectures.registry import is_scoring_model
from app.architectures.score_head import ScoreHead
from app.gguf.reader import GGUFReader
from tests.tiny_gguf_qwen3 import build_tiny_qwen3_gguf


def test_yes_no_relevance_is_the_softmax_probability_of_yes() -> None:
    labels = ["yes", "no"]
    score = ScoreHead.relevance(torch.tensor([2.0, -1.0]), labels)
    assert score == pytest.approx(1 / (1 + math.exp(-3.0)))


def test_label_order_is_read_from_the_labels_not_assumed() -> None:
    assert ScoreHead.relevance(torch.tensor([-1.0, 2.0]), ["no", "yes"]) == pytest.approx(
        ScoreHead.relevance(torch.tensor([2.0, -1.0]), ["yes", "no"])
    )


def test_a_single_logit_head_uses_the_sigmoid() -> None:
    assert ScoreHead.relevance(torch.tensor([0.0]), ["score"]) == pytest.approx(0.5)


def test_an_unknown_multi_label_head_is_refused() -> None:
    with pytest.raises(ValueError):
        ScoreHead.relevance(torch.tensor([1.0, 2.0, 3.0]), ["a", "b", "c"])


def test_a_reranker_gguf_is_detected_and_a_chat_gguf_is_not(tmp_path) -> None:
    reranker = build_tiny_qwen3_gguf(tmp_path / "rerank.gguf", score_labels=["yes", "no"])
    chat = build_tiny_qwen3_gguf(tmp_path / "chat.gguf", tied_embeddings=True)

    assert is_scoring_model(reranker) is True
    assert is_scoring_model(chat) is False
    assert ScoreHead.labels(GGUFReader(reranker).read().metadata) == ["yes", "no"]
