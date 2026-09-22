#!/usr/bin/env bash
# Stops a matricxon instance started via scripts/start.sh.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PID_FILE="run/matricxon.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "matricxon is not running (no pid file)"
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "matricxon is not running (stale pid file, removing it)"
    rm -f "$PID_FILE"
    exit 0
fi

kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 20); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "matricxon stopped (pid $PID)"
        exit 0
    fi
    sleep 0.5
done

echo "matricxon did not stop gracefully, sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
