import regex

from app.gguf.metadata import GGUFMetadata

_CONTROL_TOKEN_TYPE = 3
_USER_DEFINED_TOKEN_TYPE = 4
_BYTE_TOKEN_TYPE = 6
_SPACE_MARKER = "▁"


class Gemma4Tokenizer:
    """BPE tokenizer built from a GGUF's `tokenizer.ggml.*` metadata for `tokenizer.ggml.model =
    "gemma4"` (checked on a `google/gemma-4-12b-it` pull). Rank-based merge-list BPE
    (`tokenizer.ggml.merges`, as `GGUFTokenizer` for "gpt2"), but that class can't be reused:
    Gemma's vocab keeps ordinary Unicode characters instead of GPT-2's byte-to-unicode remapping
    (falling back to literal `<0xNN>` byte tokens only for out-of-vocab pieces, like
    `SentencePieceTokenizer` but with rank-based, not score-based, merges), and there is no GPT-2
    regex pre-tokenizer: the whole input is merged as one span (this GGUF has no
    `tokenizer.ggml.pre`).

    Reverse-engineered from the HF `tokenizer.json` (`bt.pre_tokenizer`/`bt.normalizer`) and
    cross-checked end to end (tools/oracle/validate_gemma_tokenizer.py). Two earlier hypotheses
    failed on real prompts (GPT-2-style word-then-merge; attaching the space marker to the preceding
    word) before the rule was settled: normalize the whole text with no dummy prefix, merge it as
    one span.
    """

    def __init__(self, metadata: GGUFMetadata) -> None:
        tokens: list[str] = metadata.require("tokenizer.ggml.tokens")
        merges: list[str] = metadata.require("tokenizer.ggml.merges")
        token_types: list[int] = metadata.require("tokenizer.ggml.token_type")

        self._token_to_id = {token: i for i, token in enumerate(tokens)}
        self._id_to_token = tokens
        self._token_types = token_types
        self._byte_token_ids = {
            int(token[3:5], 16): i
            for i, (token, token_type) in enumerate(zip(tokens, token_types, strict=True))
            if token_type == _BYTE_TOKEN_TYPE
        }
        self._bpe_ranks = {tuple(pair.split(" ", 1)): rank for rank, pair in enumerate(merges)}
        self._bpe_cache: dict[str, list[str]] = {}
        self._control_tokens = regex.compile(
            "|".join(
                regex.escape(token)
                for token, token_type in zip(tokens, token_types, strict=True)
                if token_type in (_CONTROL_TOKEN_TYPE, _USER_DEFINED_TOKEN_TYPE)
            )
        )

        self.bos_token_id: int | None = metadata.get("tokenizer.ggml.bos_token_id")
        self.eos_token_id: int | None = metadata.get("tokenizer.ggml.eos_token_id")

    def encode(self, text: str, add_bos: bool = False) -> list[int]:
        """`add_bos` is the caller's explicit call, as
        `GGUFTokenizer.encode`/`SentencePieceTokenizer.encode` (see their docstrings for why)."""
        token_ids: list[int] = []
        if add_bos and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)

        for span in self._split_on_control_tokens(text):
            if span in self._token_to_id and self._control_tokens.fullmatch(span):
                token_ids.append(self._token_to_id[span])
                continue
            normalized = span.replace(" ", _SPACE_MARKER)
            for symbol in self._bpe(normalized):
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

    def _symbol_to_token_ids(self, symbol: str) -> list[int]:
        token_id = self._token_to_id.get(symbol)
        if token_id is not None:
            return [token_id]
        return [self._byte_token_ids[b] for b in symbol.encode("utf-8")]

    def _bpe(self, word: str) -> list[str]:
        """Rank-based merge selection (lowest merge rank wins) over the *whole* normalized span at
        once (class docstring says why no regex pre-split). Same algorithm/complexity as
        `GGUFTokenizer._bpe`, without its byte-to-unicode remapping."""
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

    def _split_on_control_tokens(self, text: str) -> list[str]:
        pieces = self._control_tokens.split(text)
        matches = self._control_tokens.findall(text)
        interleaved = [None] * (len(pieces) + len(matches))
        interleaved[0::2] = pieces
        interleaved[1::2] = matches
        return [piece for piece in interleaved if piece]
