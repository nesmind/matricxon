"""Cross-checks GGUFTokenizer's encode() against the real HF tokenizer for
the same model - the encoding/decoding equivalent of scripts/run_m3_oracle_check.sh
for the forward pass. Needs no memory cap: the GGUF side only reads a KB-scale
header (never the multi-GB tensor data), and the HF side is a small
tokenizer.json fetch, not model weights.

Known, accepted mismatches (not bugs - see README_oracle.md's spirit of
distinguishing real bugs from documented limits):
  - BOS: GGUF's `tokenizer.ggml.add_bos_token=True` doesn't match this
    model's real tokenizer_config.json (`add_bos_token=False` - BOS is
    inserted by chat-template application instead, which GGUFTokenizer
    deliberately leaves to the caller). Compared here with `add_bos=False`
    on our side and `add_special_tokens=False` on HF's to make both sides
    agree on that point rather than fight over it.
  - Tabs: GGUF's `tokenizer.ggml.pre="default"` (GPT-2-style regex) doesn't
    exactly replicate this tekken-derived tokenizer's whitespace handling
    around literal tab characters. Rare in real chat prompts; flagged, not
    fixed, until it actually matters.
"""

from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.tokenizer import GGUFTokenizer
from scripts.oracle.common import DEFAULT_GGUF_PATH, HF_REPO

_TEST_PROMPTS = [
    "The capital of France is",
    "Hello, world! This is a test.",
    "Numbers: 12345 and 6.789, punctuation!! -- test.",
    "[INST] What is 2+2? [/INST]",
    "Emoji test: \U0001f389 café naïve résumé",
    "   leading and trailing spaces   ",
    "A longer paragraph with various punctuation: commas, periods. "
    'Question marks? Exclamation!! And "quotes" and (parentheses).',
    "Repeated    spaces     test",
]


def main() -> None:
    parsed = GGUFReader(Path(DEFAULT_GGUF_PATH)).read()
    ours = GGUFTokenizer(parsed.metadata)
    hf = AutoTokenizer.from_pretrained(HF_REPO)

    all_ok = True
    for prompt in _TEST_PROMPTS:
        our_ids = ours.encode(prompt)
        hf_ids = hf(prompt, add_special_tokens=False)["input_ids"]
        ok = our_ids == hf_ids
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {prompt!r}")
        if not ok:
            print(f"  ours: {our_ids}")
            print(f"  hf:   {hf_ids}")

    print(f"\n{'All prompts matched.' if all_ok else 'Mismatches found - see FAIL lines above.'}")


if __name__ == "__main__":
    main()
