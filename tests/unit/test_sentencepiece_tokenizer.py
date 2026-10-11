"""Offline tests for SentencePieceTokenizer on a small hand-built vocab. The cross-check against
TinyLlama-1.1B-Chat-v1.0's HF tokenizer is tools/oracle/validate_sentencepiece_tokenizer.py
(dev-only)."""

from app.gguf.metadata import GGUFMetadata
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer

# A small SPM vocab: control tokens, a few normal pieces (one reachable only via a score-based
# merge) and all 256 byte-fallback tokens.
_NORMAL_PIECES = [
    ("▁", 0.0),
    ("h", -1.0),
    ("i", -1.0),
    ("▁h", -0.5),
    ("▁hi", 2.0),
    ("▁w", -0.5),
]


def _build_metadata() -> GGUFMetadata:
    control = [("<unk>", -1000.0, 3), ("<s>", -1000.0, 3), ("</s>", -1000.0, 3)]
    normal = [(text, score, 1) for text, score in _NORMAL_PIECES]
    byte_tokens = [(f"<0x{b:02X}>", 0.0, 6) for b in range(256)]

    entries = control + normal + byte_tokens
    tokens = [e[0] for e in entries]
    scores = [e[1] for e in entries]
    token_types = [e[2] for e in entries]

    return GGUFMetadata(
        {
            "tokenizer.ggml.tokens": tokens,
            "tokenizer.ggml.scores": scores,
            "tokenizer.ggml.token_type": token_types,
            "tokenizer.ggml.bos_token_id": 1,
            "tokenizer.ggml.eos_token_id": 2,
        }
    )


class TestEncode:
    def test_prefers_the_highest_scoring_merge(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        ids = tokenizer.encode("hi")

        assert [tokenizer._id_to_token[i] for i in ids] == ["▁hi"]

    def test_adds_dummy_prefix_only_once_at_the_start(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        ids = tokenizer.encode("hi<s>hi")

        pieces = [tokenizer._id_to_token[i] for i in ids]
        # The first "hi" merges into the dummy-prefixed "▁hi"; the second has no prefix, so it falls
        # back to single-character pieces.
        assert pieces == ["▁hi", "<s>", "h", "i"]

    def test_leading_space_normalizes_the_same_as_no_leading_space(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        assert tokenizer.encode("hi") == tokenizer.encode(" hi")

    def test_add_bos_prepends_the_bos_token_id(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        ids = tokenizer.encode("hi", add_bos=True)

        assert ids[0] == 1

    def test_unmergeable_character_falls_back_to_utf8_bytes(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        # Encoded after a control token, so the dummy prefix (first span only) adds no extra "▁".
        ids = tokenizer.encode("<s>z")

        assert [tokenizer._id_to_token[i] for i in ids][1:] == [f"<0x{ord('z'):02X}>"]

    def test_multibyte_character_falls_back_to_multiple_byte_tokens(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        ids = tokenizer.encode("<s>é")
        expected_bytes = "é".encode()

        assert len(ids) - 1 == len(expected_bytes)
        assert [tokenizer._id_to_token[i] for i in ids][1:] == [
            f"<0x{b:02X}>" for b in expected_bytes
        ]

    def test_empty_string_encodes_to_nothing(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())

        assert tokenizer.encode("") == []


class TestDecode:
    def test_normal_piece_converts_space_marker_back_to_a_real_space(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())
        ids = tokenizer.encode("hi")

        assert tokenizer.decode(ids) == " hi"

    def test_byte_fallback_round_trips_exactly(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())
        text = "zé"
        # Avoids the dummy prefix, as the encode-side byte-fallback tests, to isolate the round
        # trip.
        ids = tokenizer.encode("<s>" + text)

        assert tokenizer.decode(ids) == "<s>" + text

    def test_token_to_bytes_of_a_byte_token_returns_that_single_byte(self) -> None:
        tokenizer = SentencePieceTokenizer(_build_metadata())
        token_id = tokenizer._token_to_id["<0x41>"]

        assert tokenizer.token_to_bytes(token_id) == b"A"
