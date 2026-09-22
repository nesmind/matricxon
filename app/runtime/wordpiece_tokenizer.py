import unicodedata

from app.gguf.metadata import GGUFMetadata

_MAX_CHARS_PER_WORD = 100


class WordPieceTokenizer:
    """WordPiece tokenizer for GGUF's `tokenizer.ggml.model = "bert"` vocab
    format - used by both `bert` and `nomic-bert` arch GGUF files (confirmed
    against the real `all-MiniLM-L6-v2`/`nomic-embed-text-v1.5` GGUFs, not
    assumed).

    llama.cpp's converter rewrites the classic HF WordPiece vocab into a
    SentencePiece-style representation: a word-initial piece is stored with
    a leading `"▁"` (the same marker SentencePiece itself uses), and a
    continuation piece is stored bare - the *opposite* of the original
    vocab.txt convention, where the initial piece is bare and continuations
    carry a `"##"` prefix. Verified byte-for-byte against the real HF
    tokenizer: HF's `"hello"`/`"##ing"` land on the exact same ids as this
    vocab's `"▁hello"`/`"ing"`.

    v1 always lowercases (`do_lower_case=True`) - both real local fixtures
    use an uncased vocab, and GGUF carries no explicit cased/uncased flag
    to branch on.
    """

    _WORD_INITIAL_MARKER = "▁"

    def __init__(self, metadata: GGUFMetadata) -> None:
        tokens: list[str] = metadata.require("tokenizer.ggml.tokens")
        self._token_to_id = {token: i for i, token in enumerate(tokens)}
        self._id_to_token = tokens

        # cls_token_id/seperator_token_id aren't always present - confirmed against two real
        # nomic-embed-text-v1.5 GGUF pulls that otherwise look identical (same tensor/kv count):
        # one had explicit cls_token_id/mask_token_id, one didn't. llama.cpp's own bert.cpp treats
        # bos_token_id/eos_token_id as the CLS/SEP aliases in exactly this situation, so this
        # mirrors that fallback rather than hard-failing on a spec-valid file.
        cls_token_id = metadata.get("tokenizer.ggml.cls_token_id")
        self.cls_token_id: int = (
            cls_token_id
            if cls_token_id is not None
            else metadata.require("tokenizer.ggml.bos_token_id")
        )
        sep_token_id = metadata.get("tokenizer.ggml.seperator_token_id")
        self.sep_token_id: int = (
            sep_token_id
            if sep_token_id is not None
            else metadata.require("tokenizer.ggml.eos_token_id")
        )
        self.unk_token_id: int = metadata.require("tokenizer.ggml.unknown_token_id")

    def encode(self, text: str) -> list[int]:
        """Returns `[CLS], <wordpiece ids>, [SEP]` - the standard BERT
        single-sequence input shape (v1 has no need for sentence-pair
        `[SEP]`-joined inputs)."""
        token_ids = [self.cls_token_id]
        for word in self._basic_tokenize(text):
            token_ids.extend(self._wordpiece_tokenize(word))
        token_ids.append(self.sep_token_id)
        return token_ids

    def _basic_tokenize(self, text: str) -> list[str]:
        text = self._clean_text(text)
        words: list[str] = []
        for whitespace_chunk in text.split():
            words.extend(self._split_on_punctuation(self._strip_accents(whitespace_chunk.lower())))
        return words

    def _clean_text(self, text: str) -> str:
        chars = []
        for char in text:
            codepoint = ord(char)
            if codepoint == 0 or codepoint == 0xFFFD or unicodedata.category(char) == "Cc":
                continue
            chars.append(" " if unicodedata.category(char) == "Zs" else char)
        return "".join(chars)

    def _strip_accents(self, text: str) -> str:
        decomposed = unicodedata.normalize("NFD", text)
        return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")

    def _split_on_punctuation(self, word: str) -> list[str]:
        pieces: list[list[str]] = []
        start_new_piece = True
        for char in word:
            if self._is_punctuation(char):
                pieces.append([char])
                start_new_piece = True
            else:
                if start_new_piece:
                    pieces.append([])
                pieces[-1].append(char)
                start_new_piece = False
        return ["".join(piece) for piece in pieces if piece]

    @staticmethod
    def _is_punctuation(char: str) -> bool:
        codepoint = ord(char)
        ascii_punctuation_ranges = (
            (33, 47),  # ! " # $ % & ' ( ) * + , - . /
            (58, 64),  # : ; < = > ? @
            (91, 96),  # [ \ ] ^ _ `
            (123, 126),  # { | } ~
        )
        if any(lo <= codepoint <= hi for lo, hi in ascii_punctuation_ranges):
            return True
        return unicodedata.category(char).startswith("P")

    def _wordpiece_tokenize(self, word: str) -> list[int]:
        if len(word) > _MAX_CHARS_PER_WORD:
            return [self.unk_token_id]

        token_ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            matched_id: int | None = None
            while start < end:
                candidate = word[start:end]
                lookup = (self._WORD_INITIAL_MARKER + candidate) if start == 0 else candidate
                if lookup in self._token_to_id:
                    matched_id = self._token_to_id[lookup]
                    break
                end -= 1
            if matched_id is None:
                return [self.unk_token_id]
            token_ids.append(matched_id)
            start = end
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Debug/inspection convenience - EmbeddingEngine never decodes its
        own output, only encode() is on the real request path."""
        words: list[str] = []
        for token_id in token_ids:
            piece = self._id_to_token[token_id]
            if piece.startswith(self._WORD_INITIAL_MARKER):
                words.append(piece[len(self._WORD_INITIAL_MARKER) :])
            elif words:
                words[-1] += piece
            else:
                words.append(piece)
        return " ".join(words)
