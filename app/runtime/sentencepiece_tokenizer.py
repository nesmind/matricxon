import regex

from app.gguf.metadata import GGUFMetadata

_CONTROL_TOKEN_TYPE = 3
_BYTE_TOKEN_TYPE = 6
_SPACE_MARKER = "▁"  # "▁" - SentencePiece's word-boundary/space marker


class SentencePieceTokenizer:
    """SentencePiece BPE tokenizer built directly from a GGUF file's

    `tokenizer.ggml.*` metadata - matches `tokenizer.ggml.model = "llama"`
    (confirmed against a real `TinyLlama-1.1B-Chat-v1.0` GGUF pull, not
    assumed), llama.cpp's `LLAMA_VOCAB_TYPE_SPM`. A genuinely different core
    algorithm from `GGUFTokenizer`'s byte-level BPE (M5.5): no
    `tokenizer.ggml.merges` key exists for this vocab type at all - merge
    priority instead comes from a per-vocab-entry `tokenizer.ggml.scores`
    float (higher score merges first), and the input text is normalized as
    one whole span (spaces become "▁", plus one leading "▁" unless the text
    already starts with one after that substitution - real SentencePiece's
    `add_dummy_prefix`) rather than GPT-2-regex-split into words first.
    Confirmed via a real HF `LlamaTokenizer` cross-check
    (scripts/oracle/validate_sentencepiece_tokenizer.py), not re-derived
    from memory alone - a subtly wrong merge-selection rule here would
    silently mistokenize instead of crashing.

    Falls back to per-UTF-8-byte tokens (`<0x0A>` etc., type 6 in
    `tokenizer.ggml.token_type` - 256 of them, always present for this
    vocab type) for any character with no direct vocab piece - by
    construction the only symbols that can ever need this are the
    *original*, never-merged single characters: anything produced by a
    successful merge step was, by definition, already confirmed to be a
    real vocab entry at merge time.
    """

    def __init__(self, metadata: GGUFMetadata) -> None:
        tokens: list[str] = metadata.require("tokenizer.ggml.tokens")
        scores: list[float] = metadata.require("tokenizer.ggml.scores")
        token_types: list[int] = metadata.require("tokenizer.ggml.token_type")

        self._token_to_id = {token: i for i, token in enumerate(tokens)}
        self._id_to_token = tokens
        self._scores = scores
        self._token_types = token_types
        self._byte_token_ids = {
            int(token[3:5], 16): i
            for i, (token, token_type) in enumerate(zip(tokens, token_types, strict=True))
            if token_type == _BYTE_TOKEN_TYPE
        }
        self._control_tokens = regex.compile(
            "|".join(
                regex.escape(token)
                for token, token_type in zip(tokens, token_types, strict=True)
                if token_type == _CONTROL_TOKEN_TYPE
            )
        )

        self.bos_token_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_token_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """`add_bos` is the caller's explicit call, same contract as

        `GGUFTokenizer.encode` - see that method's own docstring for why.
        """
        token_ids: list[int] = []
        if add_bos and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        spans = self._split_on_control_tokens(text)
        for index, span in enumerate(spans):
            if span in self._token_to_id and self._control_tokens.fullmatch(span):
                token_ids.append(self._token_to_id[span])
                continue
            # Real SentencePiece's add_dummy_prefix applies once, to the very first
            # fragment of the whole call - never to a fragment that follows a literal
            # control token elsewhere in the text (confirmed against the real HF
            # tokenizer: "hello<s>world" -> ["▁hello", "<s>", "world"], not "▁world").
            normalized = self._normalize(span, add_dummy_prefix=(index == 0))
            for symbol in self._merge_piece(normalized):
                token_ids.extend(self._symbol_to_token_ids(symbol))

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        raw_bytes = b"".join(self.token_to_bytes(i) for i in token_ids)
        return raw_bytes.decode("utf-8", errors="replace")

    def token_to_bytes(self, token_id: int) -> bytes:
        text = self._id_to_token[token_id]
        if self._token_types[token_id] == _BYTE_TOKEN_TYPE:
            return bytes([int(text[3:5], 16)])
        return text.replace(_SPACE_MARKER, " ").encode("utf-8")

    def _normalize(self, text: str, add_dummy_prefix: bool) -> str:
        replaced = text.replace(" ", _SPACE_MARKER)
        if add_dummy_prefix and not replaced.startswith(_SPACE_MARKER):
            replaced = _SPACE_MARKER + replaced
        return replaced

    def _merge_piece(self, normalized: str) -> list[str]:
        """Repeatedly merges the adjacent symbol pair whose concatenation is

        a real vocab entry with the highest score, until no valid merge
        remains - real SentencePiece BPE's actual selection rule (score,
        not an explicit ordered merge-list like GPT-2's). O(n^2) rescan per
        merge, same complexity class as `GGUFTokenizer._bpe`'s equivalent
        loop - fine at real chat-message lengths, not a hot path.
        """
        symbols = list(normalized)
        while len(symbols) > 1:
            best_index = -1
            best_score = float("-inf")
            for i in range(len(symbols) - 1):
                token_id = self._token_to_id.get(symbols[i] + symbols[i + 1])
                if token_id is None:
                    continue
                score = self._scores[token_id]
                if score > best_score:
                    best_score = score
                    best_index = i
            if best_index == -1:
                break
            symbols[best_index : best_index + 2] = [symbols[best_index] + symbols[best_index + 1]]
        return symbols

    def _symbol_to_token_ids(self, symbol: str) -> list[int]:
        token_id = self._token_to_id.get(symbol)
        if token_id is not None:
            return [token_id]
        return [self._byte_token_ids[b] for b in symbol.encode("utf-8")]

    def _split_on_control_tokens(self, text: str) -> list[str]:
        pieces = self._control_tokens.split(text)
        matches = self._control_tokens.findall(text)
        interleaved = [None] * (len(pieces) + len(matches))
        interleaved[0::2] = pieces
        interleaved[1::2] = matches
        return [piece for piece in interleaved if piece]
