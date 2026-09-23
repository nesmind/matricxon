#!/usr/bin/env bash
# Installs matricxon on macOS, in this checkout: the manual steps from README.md's
# "Setup on macOS", automated. Written for macOS's own bash 3.2 (no bash-4-only features).
# pAIring's "Install from GitHub" button only runs on Linux, which is why this exists.
#
# Usage: scripts/install_mac.sh [--yes]
#   --yes   install Homebrew's libomp without asking (multi-threaded native kernels)
#
# NOT yet tested on a real Mac - please report its output if anything fails.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

ASSUME_YES=0
[[ "${1:-}" == "--yes" ]] && ASSUME_YES=1

step() { printf '\n==> %s\n' "$1"; }
fail() { printf '\nERROR: %s\n' "$1" >&2; exit 1; }

if [[ "$(uname -s)" != "Darwin" ]]; then
    fail "this script is for macOS. On Linux, use pAIring's \"Install from GitHub\" or README.md's Setup."
fi

step "Checking Xcode command-line tools (git + the cc C compiler)"
if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install || true
    fail "Xcode command-line tools are being installed - finish that dialog, then run this script again."
fi
command -v git >/dev/null || fail "git not found even though the command-line tools are installed."
command -v cc >/dev/null || fail "cc not found even though the command-line tools are installed."
echo "ok: $(cc --version | head -1)"

step "Finding Python 3.11 or newer"
PYTHON=""
for candidate in "${PYTHON_BIN:-}" python3.13 python3.12 python3.11 python3; do
    [[ -z "$candidate" ]] && continue
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
[[ -n "$PYTHON" ]] || fail "no Python 3.11+ found. Install one with: brew install python@3.13"
echo "ok: $("$PYTHON" --version) ($(command -v "$PYTHON"))"

step "Checking Homebrew's libomp (lets the native kernels use several CPU cores)"
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found - skipping. The native kernels will build single-threaded."
elif [[ -d "$(brew --prefix)/opt/libomp/lib" ]]; then
    echo "ok: libomp already installed"
else
    answer="n"
    if [[ "$ASSUME_YES" == "1" ]]; then
        answer="y"
    elif [[ -t 0 ]]; then
        read -r -p "libomp isn't installed. Install it now with Homebrew? [Y/n] " answer
        answer="${answer:-y}"
    fi
    if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
        brew install libomp
    else
        echo "Skipped - the native kernels will build single-threaded (brew install libomp later to change that)."
    fi
fi

step "Creating the Python environment (.venv)"
if [[ -x .venv/bin/python ]]; then
    echo "ok: .venv already exists - reusing it"
else
    "$PYTHON" -m venv .venv
fi

step "Installing dependencies (PyTorch is a large download - this can take several minutes)"
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

step "Writing .env"
if [[ -f .env ]]; then
    echo "ok: .env already exists - left unchanged"
else
    cat > .env <<'EOF'
# matricxon settings - see app/config.py for every option. Real environment variables win over this file.
MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE=true
MATRICXON_GEMV_BACKEND=native
MATRICXON_LOG_LEVEL=1
# Where models are stored (default: ./data/models inside this folder):
# MATRICXON_MODELS_DIR=/Users/you/matricxon-models
# CPU threads (default: every core) - fewer runs cooler:
# MATRICXON_TORCH_THREADS=4
EOF
    echo "ok: created .env (native kernels on, log level 1)"
fi

step "Building the native C kernels"
PYTHONPATH="$PROJECT_DIR" .venv/bin/python - <<'EOF'
from app.native.gemm import NativeGemm
from app.native.library import NativeBuildError, NativeKernelLibrary

try:
    library = NativeKernelLibrary().load()
except NativeBuildError as exc:
    print(f"The native kernels did not build - matricxon will use its Numba kernels instead.\n{exc}")
else:
    info = NativeGemm(library, 1).build_info
    threading = "multi-threaded" if "openmp=yes" in info else "single-threaded"
    print(f"ok: native kernels built ({threading}; {info})")
EOF

step "Done"
cat <<EOF
Start matricxon:   scripts/start.sh      (then: scripts/status.sh, scripts/stop.sh)
Log file:          logs/matricxon.log
Port:              8420

To use it from pAIring: Settings > External servers > Matricxon - set "Project directory" to
  $PROJECT_DIR
or, if pAIring runs on another machine, switch to Remote mode and add http://<this-mac>:8420
EOF
