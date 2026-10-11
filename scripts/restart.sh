#!/usr/bin/env bash
# Restarts matricxon (Linux and macOS): stop.sh if running, then start.sh. On Debian/Ubuntu prefer
# `systemctl restart` with scripts/matricxon.service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/stop.sh"
"$SCRIPT_DIR/start.sh"
