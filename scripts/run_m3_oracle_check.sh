#!/usr/bin/env bash
# Runs the M3 HF-oracle cross-check (see tools/README_oracle.md).
#
# Each dump step runs in its own systemd user scope with a hard MemoryMax, so a runaway process is
# OOM-killed in its own cgroup instead of starving the desktop. Override with ORACLE_MEMORY_MAX.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MEMORY_MAX="${ORACLE_MEMORY_MAX:-4G}"
PYTHON=".venv/bin/python"

run_capped() {
  systemd-run --user --scope -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 -- "$@"
}

echo "== Dumping matricxon activations (capped at $MEMORY_MAX) =="
run_capped "$PYTHON" -m tools.oracle.dump_matricxon "$@"

echo "== Dumping HF oracle activations (capped at $MEMORY_MAX) =="
run_capped "$PYTHON" -m tools.oracle.dump_hf_oracle "$@"

echo "== Comparing =="
"$PYTHON" -m tools.oracle.compare
