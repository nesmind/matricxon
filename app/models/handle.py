from dataclasses import dataclass

from app.architectures.base import ModelArchitecture
from app.models.worker import ModelWorker
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer
from app.runtime.tokenizer import GGUFTokenizer
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

# The tokenizer implementations a GGUF's `tokenizer.ggml.model` can resolve
# to (see app.models.tokenizer_dispatch.build_tokenizer): byte-level BPE for
# "gpt2", WordPiece for "bert", SentencePiece BPE for "llama" (M10),
# Gemma4's own rank-based BPE for "gemma4" (M10). They share no common base
# class (their real capabilities genuinely differ - only the three decoder
# tokenizers have an eos_token_id, for instance), so callers narrow by the
# endpoint they're serving rather than a forced shared interface.
Tokenizer = GGUFTokenizer | WordPieceTokenizer | SentencePieceTokenizer | Gemma4Tokenizer


@dataclass
class ModelHandle:
    """One resident (loaded-into-memory) model.

    `last_used_at` is monotonic-clock time (whatever `ModelManager`'s
    injected `clock` returns), never wall-clock - it's only ever compared
    against another call to that same clock for eviction, and the caller
    (not this class) decides what "now" is, so eviction logic stays
    deterministically testable with a fake clock. `/api/ps`'s ISO-8601
    `expires_at` is computed from `expires_in()` at the API boundary
    instead, in ps_router.py.
    """

    tag: str
    architecture: ModelArchitecture
    tokenizer: Tokenizer
    worker: ModelWorker
    capabilities: list[str]
    size_bytes: int
    keep_alive_seconds: int
    last_used_at: float

    @property
    def expires_at(self) -> float:
        return self.last_used_at + self.keep_alive_seconds

    def expires_in(self, now: float) -> float:
        return self.expires_at - now

    def is_expired(self, now: float) -> bool:
        return self.expires_in(now) <= 0

    def touch(self, now: float, keep_alive_seconds: int | None = None) -> None:
        if keep_alive_seconds is not None:
            self.keep_alive_seconds = keep_alive_seconds
        self.last_used_at = now
