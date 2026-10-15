"""GGUFTokenizer picks its pre-tokenizer split pattern from `tokenizer.ggml.pre`
(app/runtime/tokenizer.py). Token counts for Llama-3.2 with the "llama-bpe" pattern were checked
against Ollama's prompt_eval_count on newline runs, digit runs, contractions and non-ASCII text."""

from app.gguf.metadata import GGUFMetadata
from app.runtime.tokenizer import GGUFTokenizer


def _tokenizer(pre: str | None) -> GGUFTokenizer:
    # Single-byte tokens only, no merges: every chunk encodes to one token per byte, so the token
    # list shows exactly how the pre-tokenizer split the text.
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


def test_qwen2_keeps_punctuation_with_its_newline_and_splits_single_digits() -> None:
    """Checked against llama.cpp on a Qwen3-Reranker GGUF; under GPT-2's pattern its prompt
    split "?\\n" and "<Document" apart."""
    pattern = _tokenizer("qwen2")._split_pattern

    assert pattern.findall("it?\n<Document>: 2026") == [
        "it",
        "?\n",
        "<Document",
        ">:",
        " ",
        "2",
        "0",
        "2",
        "6",
    ]


def test_user_defined_tokens_are_matched_whole_like_control_tokens() -> None:
    """Qwen's `<think>` is user-defined (type 4), not control (3); llama.cpp keeps it whole."""
    tokens = [chr(c) for c in range(ord("!"), ord("~") + 1)] + ["<think>", "<|im_end|>"]
    types = [1] * (len(tokens) - 2) + [4, 3]
    tokenizer = GGUFTokenizer(
        GGUFMetadata(
            {
                "tokenizer.ggml.tokens": tokens,
                "tokenizer.ggml.merges": [],
                "tokenizer.ggml.token_type": types,
                "tokenizer.ggml.pre": "qwen2",
            }
        )
    )

    ids = tokenizer.encode("<think>a<|im_end|>")

    assert ids == [tokens.index("<think>"), tokens.index("a"), tokens.index("<|im_end|>")]
