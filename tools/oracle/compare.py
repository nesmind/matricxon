"""Compares matricxon's and the HF oracle's dumped per-layer activations. Loads only small dumps, so
no memory cap is needed (see tools/README_oracle.md)."""

import argparse
import sys

import torch
from safetensors.torch import load_file

from tools.oracle.common import DUMP_DIR

# matricxon uses Q4_K/Q6_K and the oracle fp8/bf16, so both differ from the fp32 original; no
# bit-exact match is expected. 0.98 is far above the ~0.03-0.76 a real bug (e.g. a mispermuted
# weight) gives and below the ~0.987-0.999 measured.
COSINE_THRESHOLD = 0.98


class ActivationComparison:
    def __init__(self, name: str, matricxon: torch.Tensor, oracle: torch.Tensor) -> None:
        self.name = name
        self.cosine = torch.nn.functional.cosine_similarity(
            matricxon.flatten(), oracle.flatten(), dim=0
        ).item()
        self.max_abs_diff = (matricxon - oracle).abs().max().item()

    @property
    def passed(self) -> bool:
        return self.cosine >= COSINE_THRESHOLD

    def report_line(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.name:<12} "
            f"cosine={self.cosine:.6f}  max_abs_diff={self.max_abs_diff:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matricxon", default=str(DUMP_DIR / "matricxon_activations.safetensors"))
    parser.add_argument("--oracle", default=str(DUMP_DIR / "hf_oracle_activations.safetensors"))
    args = parser.parse_args()

    matricxon = load_file(args.matricxon)
    oracle = load_file(args.oracle)

    shared_keys = sorted(set(matricxon) & set(oracle), key=_sort_key)
    if not shared_keys:
        print("No shared tensor names between the two dumps - nothing to compare.")
        return 1

    comparisons = [ActivationComparison(key, matricxon[key], oracle[key]) for key in shared_keys]
    for comparison in comparisons:
        print(comparison.report_line())

    if all(c.passed for c in comparisons):
        print(f"\nAll {len(comparisons)} tensors matched (cosine >= {COSINE_THRESHOLD}).")
        return 0
    print("\nMismatch detected - see FAIL lines above.")
    return 1


_KIND_ORDER = {"attn": 0, "mlp": 1, "layer": 2}


def _sort_key(name: str) -> tuple[int, int, str]:
    kind, _, index = name.partition("_")
    if kind in _KIND_ORDER and index.isdigit():
        return (int(index), _KIND_ORDER[kind], name)
    return (10_000, 0, name)


if __name__ == "__main__":
    sys.exit(main())
