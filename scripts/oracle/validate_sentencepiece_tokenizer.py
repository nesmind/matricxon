"""Cross-checks SentencePieceTokenizer's encode() against the real HF

`LlamaTokenizer`/`LlamaTokenizerFast` for TinyLlama-1.1B-Chat-v1.0 - the M10
plain-`llama`-architecture equivalent of scripts/oracle/validate_tokenizer.py
(M5.5). Needs no memory cap: the GGUF side only reads a KB-scale header, the
HF side only a small tokenizer.json fetch, never model weights.

    .venv/bin/python -m scripts.oracle.validate_sentencepiece_tokenizer
"""

from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.sentencepiece_tokenizer import SentencePieceTokenizer

LLAMA_GGUF_PATH = Path("data/models/hf.co/hieupt/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF/Q4_K_M.gguf")
LLAMA_HF_REPO = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

_TEST_PROMPTS = [
    "The capital of France is",
    "hello world",
    " hello",
    "  hello",
    "hello  world",
    "hello\nworld",
    "Hello, world! This is a test.",
    "Numbers: 12345 and 6.789, punctuation!! -- test.",
    "<s>What is 2+2?</s>",
    "Emoji test: \U0001f389 café naïve résumé",
    "   leading and trailing spaces   ",
    "A longer paragraph with various punctuation: commas, periods. "
    'Question marks? Exclamation!! And "quotes" and (parentheses).',
    "Repeated    spaces     test",
    "def foo(x):\n    return x + 1\n",
    "",
    "<s>What is 2+2?</s>",
    "hello<s>world",
    "hello world<s>next span",
    "<unk>hi",
]


def main() -> None:
    metadata = GGUFReader(LLAMA_GGUF_PATH).read().metadata
    ours = SentencePieceTokenizer(metadata)
    hf = AutoTokenizer.from_pretrained(LLAMA_HF_REPO)

    all_ok = True
    for prompt in _TEST_PROMPTS:
        our_ids = ours.encode(prompt)
        hf_ids = hf.encode(prompt, add_special_tokens=False)
        ok = our_ids == hf_ids
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  ours: {our_ids}")
            print(f"  hf:   {hf_ids}")
            print(f"  ours pieces: {[ours._id_to_token[i] for i in our_ids]}")
            print(f"  hf pieces:   {hf.convert_ids_to_tokens(hf_ids)}")

    print(f"\n{'All prompts matched.' if all_ok else 'Mismatches found - see FAIL lines above.'}")


if __name__ == "__main__":
    main()
