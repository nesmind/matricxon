#!/usr/bin/env bash
# Restarts matricxon (works on Linux and macOS): stops it via scripts/stop.sh if it's running,
# then starts it again via scripts/start.sh. See those two scripts for what each step does on its
# own; for Debian/Ubuntu, prefer `systemctl restart` against scripts/matricxon.service instead,
# same as start.sh/stop.sh's own recommendation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stop.sh"
"$SCRIPT_DIR/start.sh"
