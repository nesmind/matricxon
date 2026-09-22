import regex

from app.gguf.metadata import GGUFMetadata

_CONTROL_TOKEN_TYPE = 3
_USER_DEFINED_TOKEN_TYPE = 4
_BYTE_TOKEN_TYPE = 6
_SPACE_MARKER = "▁"  # "▁"


class Gemma4Tokenizer:
    """BPE tokenizer built directly from a GGUF file's `tokenizer.ggml.*`

    metadata - matches `tokenizer.ggml.model = "gemma4"` (confirmed against
    a real `google/gemma-4-12b-it` GGUF pull, not assumed). Real
    ordered-merge-list BPE (`tokenizer.ggml.merges` exists, same rank-based
    algorithm `GGUFTokenizer` already implements for "gpt2"), but two real
    differences meant it couldn't just reuse that class: no GPT-2-style
    byte-to-unicode remapping (Gemma's vocab keeps ordinary Unicode
    characters as-is, falling back to literal `<0xNN>` byte tokens only for
    genuinely out-of-vocab pieces - the same convention `SentencePieceTokenizer`
    uses for "llama", just with rank-based merge selection instead of
    score-based), and no GPT-2 regex pre-tokenizer splitting text into
    words first - the *whole* input is merged as one span, relying on the
    trained vocab's own merge ranks to naturally respect word boundaries
    rather than an explicit pre-split (confirmed: this GGUF carries no
    `tokenizer.ggml.pre` value at all).

    Reverse-engineered against the real HF tokenizer's own serialized
    `tokenizer.json` (`bt.pre_tokenizer`/`bt.normalizer`, not guessed from a
    reference implementation) and cross-checked end to end
    (scripts/oracle/validate_gemma_tokenizer.py) - a naive first hypothesis
    (GPT-2-style word-then-merge, or attaching the space marker to the
    *preceding* word per the raw pre-tokenizer config's literal
    `Split(" ", MergedWithPrevious)` behavior) both failed against real
    prompts before the actual rule (normalize the whole text first with no
    dummy prefix, merge it as a single span) was confirmed to match.
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
        """`add_bos` is the caller's explicit call, same contract as

        `GGUFTokenizer.encode`/`SentencePieceTokenizer.encode` - see either
        one's own docstring for why.
        """
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
        """Rank-based merge selection (lowest merge rank wins) over the

        *whole* normalized span at once - see this class's own docstring
        for why there's no regex pre-split into words first. Same
        algorithm/complexity as `GGUFTokenizer._bpe`, just without that
        class's byte-to-unicode remapping.
        """
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
