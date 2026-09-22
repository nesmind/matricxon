import torch

from app.architectures.base import ModelArchitecture
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer


class EmbeddingEngine:
    """Single forward pass + mean-pooling into one embedding vector.

    v1 encodes exactly one prompt per call (no batching, no padding), so
    mean-pooling is a plain average over every position - there's no
    attention_mask needed to exclude padding, since there isn't any.
    Followed by L2 normalization: confirmed against both real local
    embedding models' actual `sentence-transformers` config (a
    `2_Normalize` step after mean-pooling), not assumed.
    """

    def __init__(self, architecture: ModelArchitecture, tokenizer: WordPieceTokenizer) -> None:
        self._architecture = architecture
        self._tokenizer = tokenizer

    def embed(self, text: str) -> list[float]:
        input_ids = torch.tensor([self._tokenizer.encode(text)], dtype=torch.long)
        with torch.no_grad():
            hidden_states = self._architecture.forward(input_ids)
            pooled = hidden_states.mean(dim=1)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return normalized.squeeze(0).tolist()
