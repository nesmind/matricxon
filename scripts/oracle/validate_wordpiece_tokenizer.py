"""Cross-checks WordPieceTokenizer against the real HF tokenizers for both
real local `bert`-model fixtures - the WordPiece equivalent of
validate_tokenizer.py. Needs no memory cap (KB-scale GGUF headers, small
tokenizer.json fetches, no model weights).
"""

from pathlib import Path

from transformers import AutoTokenizer

from app.gguf.reader import GGUFReader
from app.runtime.wordpiece_tokenizer import WordPieceTokenizer

_CASES = [
    (
        "/home/home/Code/Py/AI/pAIring/models/blobs/"
        "sha256-797b70c4edf85907fe0a49eb85811256f65fa0f7bf52166b147fd16be2be4662",
        "sentence-transformers/all-MiniLM-L6-v2",
        [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Running, jumped, and swimming quickly.",
            "unbelievable antidisestablishmentarianism",
            "Numbers: 12345 and 6.789, punctuation!! -- test.",
            "café naïve résumé",
            "   leading and trailing spaces   ",
            "A sentence with UPPERCASE and MixedCase Words.",
            "It's a test with apostrophes and 'quotes'.",
            "emoji test \U0001f389 and unicode ℝ",
        ],
    ),
    (
        "/home/home/Code/Py/AI/pAIring/models/blobs/"
        "sha256-970aa74c0a90ef7482477cf803618e776e173c007bf957f635f1015bfcfef0e6",
        "nomic-ai/nomic-embed-text-v1.5",
        [
            "search_query: what is the capital of France?",
            "A sentence with UPPERCASE and unbelievable words.",
            "café naïve résumé with numbers 123.456",
        ],
    ),
]


def main() -> None:
    all_ok = True
    for gguf_path, hf_repo, prompts in _CASES:
        if not Path(gguf_path).exists():
            print(f"[SKIP] {hf_repo} - real fixture not present on this machine: {gguf_path}")
            continue

        metadata = GGUFReader(Path(gguf_path)).read().metadata
        ours = WordPieceTokenizer(metadata)
        hf = AutoTokenizer.from_pretrained(hf_repo)

        for prompt in prompts:
            our_ids = ours.encode(prompt)
            hf_ids = hf(prompt)["input_ids"]
            ok = our_ids == hf_ids
            all_ok &= ok
            print(f"[{'PASS' if ok else 'FAIL'}] {hf_repo}: {prompt!r}")
            if not ok:
                print(f"  ours: {our_ids}")
                print(f"  hf:   {hf_ids}")

    print(f"\n{'All prompts matched.' if all_ok else 'Mismatches found - see FAIL lines above.'}")


if __name__ == "__main__":
    main()
