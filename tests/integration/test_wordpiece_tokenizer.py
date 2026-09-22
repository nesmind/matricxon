"""Fast/offline: only reads the real all-MiniLM GGUF's KB-scale header
(never the tensor data), so this needs no memory cap and no network -
unlike scripts/oracle/validate_wordpiece_tokenizer.py, which additionally
cross-checks against the real HF tokenizer. Skips gracefully without
pAIring's local blob store.
"""

import pytest

from app.gguf.reader import GGUFReader
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

_ALL_MINILM_BLOB = (
    "/home/home/Code/Py/AI/pAIring/models/blobs/"
    "sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662"
)


@pytest.fixture
def tokenizer() -> WordPieceTokenizer:
    from pathlib import Path

    if not Path(_ALL_MINILM_BLOB).exists():
        pytest.skip(f"real fixture not present on this machine: {_ALL_MINILM_BLOB}")
    metadata = GGUFReader(Path(_ALL_MINILM_BLOB)).read().metadata
    return WordPieceTokenizer(metadata)


class TestWordPieceTokenizer:
    def test_wraps_input_in_cls_and_sep(self, tokenizer: WordPieceTokenizer) -> None:
        ids = tokenizer.encode("hello")

        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id

    def test_known_special_token_ids(self, tokenizer: WordPieceTokenizer) -> None:
        assert tokenizer.cls_token_id == 101
        assert tokenizer.sep_token_id == 102
        assert tokenizer.unk_token_id == 100

    def test_lowercases_input(self, tokenizer: WordPieceTokenizer) -> None:
        assert tokenizer.encode("HELLO") == tokenizer.encode("hello")

    def test_splits_a_long_word_into_multiple_wordpieces(
        self, tokenizer: WordPieceTokenizer
    ) -> None:
        ids = tokenizer.encode("antidisestablishmentarianism")

        # [CLS] + at least 2 wordpieces + [SEP]
        assert len(ids) > 3

    def test_unknown_garbage_falls_back_to_unk(self, tokenizer: WordPieceTokenizer) -> None:
        # A "word" with no possible wordpiece match at any prefix length.
        ids = tokenizer.encode("\U0001f600\U0001f600\U0001f600" * 40)

        assert tokenizer.unk_token_id in ids

    def test_decode_round_trips_simple_text(self, tokenizer: WordPieceTokenizer) -> None:
        ids = tokenizer.encode("hello world")
        decoded = tokenizer.decode(ids)

        assert "hello" in decoded
        assert "world" in decoded
