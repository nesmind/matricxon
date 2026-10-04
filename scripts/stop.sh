#!/usr/bin/env bash
# Stops a matricxon instance started via scripts/start.sh.
set -euo pipefail

# Colored like uvicorn's own "INFO:" lines (plain text when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PID_FILE="run/matricxon.pid"

if [[ ! -f "$PID_FILE" ]]; then
    info "matricxon is not running (no pid file)"
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    info "matricxon is not running (stale pid file, removing it)"
    rm -f "$PID_FILE"
    exit 0
fi

kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        info "matricxon stopped (pid $PID)"
        exit 0
    fi
    sleep 0.5
done

info "matricxon did not stop gracefully, sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
