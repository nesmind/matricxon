import functools

import regex

from app.gguf.metadata import GGUFMetadata
from app.runtime.gemma_tokenizer import Gemma4Tokenizer
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer

# GPT-2's pre-tokenizer split pattern (llama.cpp's `tokenizer.ggml.pre = "default"`); needs
# \p{L}/\p{N} Unicode properties, hence the third-party `regex` package, not stdlib `re`.
_GPT2_SPLIT_PATTERN = regex.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

# Split patterns for other `tokenizer.ggml.pre` values, from llama.cpp's llama-vocab.cpp. Llama 3's
# ("llama-bpe") keeps a newline run together (`\s*[\r\n]+`) and caps digit runs at 3: under the
# GPT-2 pattern "\n\nSay" split into two "\n" tokens instead of Llama 3's single "\n\n" (found
# comparing a prompt's token count with Ollama's). Qwen's ("qwen2") is the same with single digits:
# under GPT-2's, "?\n" and "<Document" split apart (found comparing Qwen3-Reranker tokens with
# llama.cpp). Any other value keeps the GPT-2 pattern.
_SPLIT_PATTERNS_BY_PRE = {
    "llama-bpe": regex.compile(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "qwen2": regex.compile(
        r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
}

# Matched literally, never split by BPE: control (3) and user-defined (4) tokens, as llama.cpp does.
# Qwen's `<think>`/`<tool_call>` are user-defined; matching only control split them apart.
_LITERAL_TOKEN_TYPES = (3, 4)


@functools.lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    """GPT-2's byte<->unicode mapping: every byte value gets its own printable codepoint, so BPE
    merges work on ordinary strings instead of raw/unprintable bytes. The standard table from
    OpenAI's `gpt2/encoder.py` (a generic BPE technique, not llama.cpp/ggml code) used by every
    byte-level BPE tokenizer including HF's."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_points = byte_values.copy()
    next_point = 256
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_points.append(next_point)
            next_point += 1
    return dict(zip(byte_values, (chr(point) for point in unicode_points), strict=True))


class GGUFTokenizer:
    """Byte-level BPE tokenizer built from a GGUF's `tokenizer.ggml.*` metadata for
    `tokenizer.ggml.model = "gpt2"`, the format llama.cpp's converter normalizes supported models'
    tokenizers into (checked on `ministral-3:3b`). No HF `transformers` dependency (dev-only, for
    the oracle cross-check).

    Control tokens (type 3 in `tokenizer.ggml.token_type`: `<s>`, `[INST]`, `[TOOL_CALLS]`, ...) are
    matched literally and short-circuit BPE for that span, as in every BPE tokenizer: otherwise a
    literal `"[INST]"` in a prompt would be shredded into byte-level pieces instead of one token.
    """

    def __init__(self, metadata: GGUFMetadata) -> None:
        tokens: list[str] = metadata.require("tokenizer.ggml.tokens")
        merges: list[str] = metadata.require("tokenizer.ggml.merges")
        token_types: list[int] = metadata.require("tokenizer.ggml.token_type")

        self._split_pattern = _SPLIT_PATTERNS_BY_PRE.get(
            metadata.get("tokenizer.ggml.pre"), _GPT2_SPLIT_PATTERN
        )
        self._token_to_id = {token: i for i, token in enumerate(tokens)}
        self._id_to_token = tokens
        self._control_tokens = regex.compile(
            "|".join(
                regex.escape(token)
                for token, token_type in zip(tokens, token_types, strict=True)
                if token_type in _LITERAL_TOKEN_TYPES
            )
        )

        self._byte_encoder = _byte_to_unicode()
        self._byte_decoder = {char: byte for byte, char in self._byte_encoder.items()}
        self._bpe_ranks = {tuple(pair.split(" ", 1)): rank for rank, pair in enumerate(merges)}
        self._bpe_cache: dict[str, list[str]] = {}

        self.bos_token_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_token_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """`add_bos` defaults to False and is the caller's explicit call, not
        `tokenizer.ggml.add_bos_token`: that GGUF flag didn't match this model's behavior (the HF
        oracle's tokenizer_config.json sets `add_bos_token=False` for a bare encode, inserting BOS
        in chat-template application, which matricxon doesn't implement yet). Trust the caller (a
        future chat-template builder) over that flag."""
        token_ids: list[int] = []
        if add_bos and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        for span in self._split_on_control_tokens(text):
            if span in self._token_to_id and self._control_tokens.fullmatch(span):
                token_ids.append(self._token_to_id[span])
                continue
            for chunk in self._split_pattern.findall(span):
                byte_chunk = "".join(self._byte_encoder[b] for b in chunk.encode("utf-8"))
                token_ids.extend(self._token_to_id[piece] for piece in self._bpe(byte_chunk))

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = b"".join(self.token_to_bytes(i) for i in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id: int) -> bytes:
        text = self._id_to_token[token_id]
        return bytes(self._byte_decoder[char] for char in text if char in self._byte_decoder)

    def _split_on_control_tokens(self, text: str) -> list[str]:
        pieces = self._control_tokens.split(text)
        matches = self._control_tokens.findall(text)
        interleaved = [None] * (len(pieces) + len(matches))
        interleaved[0::2] = pieces
        interleaved[1::2] = matches
        return [piece for piece in interleaved if piece]

    def _bpe(self, word: str) -> list[str]:
        if word in self._bpe_cache:
            return self._bpe_cache[word]

        symbols = list(word)
        while len(symbols) > 1:
            pairs = {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}
            best_pair = min(pairs, key=lambda pair: self._bpe_ranks.get(pair, float("inf")))
            if best_pair not in self._bpe_ranks:
                break

            merged: list[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged

        self._bpe_cache[word] = symbols
        return symbols


class IncrementalTextDecoder:
    """Buffers raw bytes across a token stream and emits text only at a full UTF-8 character
    boundary: one multi-byte character (most emoji, accented letters) can span two BPE tokens, and
    decoding each in isolation would emit a stray replacement character."""

    def __init__(
        self, tokenizer: "GGUFTokenizer | SentencePieceTokenizer | Gemma4Tokenizer"
    ) -> None:
        self._tokenizer = tokenizer
        self._buffer = b""

    def push(self, token_id: int) -> str:
        self._buffer += self._tokenizer.token_to_bytes(token_id)
        try:
            text = self._buffer.decode("utf-8")
        except UnicodeDecodeError as exc:
            text = self._buffer[: exc.start].decode("utf-8")
            self._buffer = self._buffer[exc.start :]
            return text
        self._buffer = b""
        return text
