from dataclasses import dataclass, field

from app.architectures.base import ModelArchitecture
from app.models.worker import ModelWorker
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.prompt_cache import PromptCache
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
    # This model's own chat-template prompt builder (app.runtime.chat_template); None falls back
    # to Mistral3PromptBuilder in chat_router.
    prompt_builder: object | None = None
    # This model's KV cache kept between calls, so a chat's next turn only prefills its new tokens
    # (see PromptCache). Freed with the handle when the model is evicted/unloaded.
    prompt_cache: PromptCache = field(default_factory=PromptCache)
    # Real stop-token ids beyond tokenizer.eos_token_id (see app.runtime.chat_stop_tokens) -
    # computed once at load time from this model's own real metadata, empty for anything not
    # ChatML-shaped. Routers union this with {tokenizer.eos_token_id} for both ChatEngine's own
    # stop condition and hiding the closing token's text from the visible stream.
    extra_eos_token_ids: frozenset[int] = frozenset()

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
