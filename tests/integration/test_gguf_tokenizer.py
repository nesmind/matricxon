"""Fast/offline: only reads the real ministral-3:3b GGUF's KB-scale header
(never the multi-GB tensor data), so this needs no memory cap and no
network - unlike scripts/oracle/validate_tokenizer.py, which additionally
cross-checks against the real HF tokenizer as a deeper oracle. Skips
gracefully without pAIring's local blob store, matching
tests/integration/conftest.py's real_ministral_model fixture.
"""

import pytest

from app.gguf.reader import GGUFReader
from app.runtime.tokenizer import GGUFTokenizer
from tests.integration.conftest import REAL_MINISTRAL_BLOB


@pytest.fixture
def tokenizer() -> GGUFTokenizer:
    if not REAL_MINISTRAL_BLOB.exists():
        pytest.skip(f"real fixture not present on this machine: {REAL_MINISTRAL_BLOB}")
    parsed = GGUFReader(REAL_MINISTRAL_BLOB).read()
    return GGUFTokenizer(parsed.metadata)


class TestGGUFTokenizer:
    def test_round_trips_plain_text(self, tokenizer: GGUFTokenizer) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_round_trips_unicode_and_punctuation(self, tokenizer: GGUFTokenizer) -> None:
        text = 'café naïve 🎉 — em dash, and "quotes".'
        assert tokenizer.decode(tokenizer.encode(text)) == text

    def test_control_token_encodes_as_a_single_id(self, tokenizer: GGUFTokenizer) -> None:
        # Real ids for this model's tokenizer.ggml.tokens: [..., "[INST]"=3, "[/INST]"=4, ...]
        assert tokenizer.encode("[INST]") == [3]

    def test_control_tokens_short_circuit_around_ordinary_text(
        self, tokenizer: GGUFTokenizer
    ) -> None:
        ids = tokenizer.encode("[INST] hello [/INST]")
        assert ids[0] == 3
        assert ids[-1] == 4

    def test_known_special_token_ids(self, tokenizer: GGUFTokenizer) -> None:
        assert tokenizer.bos_token_id == 1
        assert tokenizer.eos_token_id == 2

    def test_add_bos_is_opt_in(self, tokenizer: GGUFTokenizer) -> None:
        without_bos = tokenizer.encode("hello")
        with_bos = tokenizer.encode("hello", add_bos=True)
        assert with_bos == [tokenizer.bos_token_id, *without_bos]
