"""GGUFTokenizer picks its pre-tokenizer split pattern from `tokenizer.ggml.pre` (see
app/runtime/tokenizer.py). Token counts for Llama-3.2 with the "llama-bpe" pattern were checked
against Ollama's own prompt_eval_count on newline runs, digit runs, contractions and non-ASCII
text (2026-09-23)."""

from app.gguf.metadata import GGUFMetadata
from app.runtime.tokenizer import GGUFTokenizer


def _tokenizer(pre: str | None) -> GGUFTokenizer:
    # Single-byte tokens only, no merges: every split chunk encodes to one token per byte, so
    # the token list shows exactly how the pre-tokenizer split the text.
    tokens = [chr(c) for c in range(ord("!"), ord("~") + 1)] + ["Ċ", "Ġ"]
    values = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.token_type": [1] * len(tokens),
    }
    if pre is not None:
        values["tokenizer.ggml.pre"] = pre
    return GGUFTokenizer(GGUFMetadata(values))


def test_llama_bpe_keeps_a_newline_run_together_and_splits_digits_by_three() -> None:
    pattern = _tokenizer("llama-bpe")._split_pattern

    assert pattern.findall("\n\nSay 12345") == ["\n\n", "Say", " ", "123", "45"]


def test_unknown_or_missing_pre_keeps_the_gpt2_pattern() -> None:
    for pre in (None, "some-future-pre"):
        pattern = _tokenizer(pre)._split_pattern
        assert pattern.findall("\n\nSay 12345") == ["\n", "\n", "Say", " 12345"]
