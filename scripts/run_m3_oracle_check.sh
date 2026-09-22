#!/usr/bin/env bash
# Runs the M3 HF-oracle cross-check (see scripts/README_oracle.md).
#
# Each dump step runs as its own systemd user scope with a hard MemoryMax, so
# a runaway process gets OOM-killed inside its own cgroup instead of starving
# the rest of the desktop (which is what crashed VS Code last time this was
# attempted by hand). Override the cap with ORACLE_MEMORY_MAX if needed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MEMORY_MAX="${ORACLE_MEMORY_MAX:-4G}"
PYTHON=".venv/bin/python"

run_capped() {
  systemd-run --user --scope -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 -- "$@"
}

echo "== Dumping matricxon activations (capped at $MEMORY_MAX) =="
run_capped "$PYTHON" -m scripts.oracle.dump_matricxon "$@"

echo "== Dumping HF oracle activations (capped at $MEMORY_MAX) =="
run_capped "$PYTHON" -m scripts.oracle.dump_hf_oracle "$@"

echo "== Comparing =="
"$PYTHON" -m scripts.oracle.compare
