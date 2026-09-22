#!/usr/bin/env bash
# Starts matricxon detached (works on Linux and macOS). See scripts/stop.sh /
# scripts/status.sh. For Debian/Ubuntu, prefer scripts/matricxon.service instead.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${MATRICXON_HOST:-0.0.0.0}"
PORT="${MATRICXON_PORT:-8420}"
PID_FILE="run/matricxon.pid"
LOG_FILE="logs/matricxon.log"

mkdir -p run logs

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "matricxon is already running (pid $(cat "$PID_FILE"))"
    exit 1
fi

rm -f "$PID_FILE"

LAUNCHER=()
command -v setsid >/dev/null 2>&1 && LAUNCHER=(setsid)

MATRICXON_PID_FILE="$PROJECT_DIR/$PID_FILE" "${LAUNCHER[@]}" .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 < /dev/null &
disown

for _ in $(seq 1 40); do
    [[ -f "$PID_FILE" ]] && break
    sleep 0.25
done

if [[ ! -f "$PID_FILE" ]]; then
    echo "matricxon failed to start within 10s — check $LOG_FILE"
    exit 1
fi

echo "matricxon started (pid $(cat "$PID_FILE")) on $HOST:$PORT, logging to $LOG_FILE"
