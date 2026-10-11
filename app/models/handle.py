from dataclasses import dataclass, field

from app.architectures.base import ModelArchitecture
from app.models.worker import ModelWorker
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.prompt_cache import PromptCache
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer
from app.runtime.tokenizer import GGUFTokenizer
from app.runtime.tool_call_formats import ToolCallFormat
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

# The tokenizers a GGUF's `tokenizer.ggml.model` can resolve to
# (app.models.tokenizer_dispatch.build_tokenizer): byte-level BPE for "gpt2", WordPiece for "bert",
# SentencePiece BPE for "llama" (M10), Gemma4's rank-based BPE for "gemma4" (M10). They share no
# base class (their capabilities differ: only the decoder tokenizers have an eos_token_id), so
# callers narrow by the endpoint they serve.
Tokenizer = GGUFTokenizer | WordPieceTokenizer | SentencePieceTokenizer | Gemma4Tokenizer


@dataclass
class ModelHandle:
    """One resident (loaded-into-memory) model.

    `last_used_at` is monotonic-clock time (whatever `ModelManager`'s injected `clock` returns),
    never wall-clock: only compared against that same clock for eviction, and the caller decides
    "now", keeping eviction deterministically testable with a fake clock. `/api/ps`'s ISO-8601
    `expires_at` comes from `expires_in` at the API boundary (ps_router.py).
    """

    tag: str
    architecture: ModelArchitecture
    tokenizer: Tokenizer
    worker: ModelWorker
    capabilities: list[str]
    size_bytes: int
    keep_alive_seconds: int
    last_used_at: float
    # This model's chat-template prompt builder (app.runtime.chat_template); None falls back to
    # Mistral3PromptBuilder in chat_router.
    prompt_builder: object | None = None
    # How this model writes a tool call in its reply (None: unknown, tools aren't parsed back).
    tool_call_format: ToolCallFormat | None = None
    # This model's KV cache kept between calls, so a chat's next turn only prefills its new tokens
    # (PromptCache). Freed with the handle on eviction/unload.
    prompt_cache: PromptCache = field(default_factory=PromptCache)
    # Stop-token ids beyond tokenizer.eos_token_id (app.runtime.chat_stop_tokens), computed at load
    # time from this model's metadata; empty for anything not ChatML-shaped. Routers union it with
    # {tokenizer.eos_token_id} for ChatEngine's stop condition and to hide the closing token's text.
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
