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

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "port $PORT is already in use - is another matricxon running (e.g. pAIring's external/matricxon)?"
    echo "Stop that one first, or start this one on another port: MATRICXON_PORT=8421 scripts/start.sh"
    exit 1
fi

if command -v setsid >/dev/null 2>&1; then
    LAUNCHER=(setsid)
else
    LAUNCHER=(nohup)
fi

MATRICXON_PID_FILE="$PROJECT_DIR/$PID_FILE" "${LAUNCHER[@]}" .venv/bin/uvicorn app.main:app --host "$HOST" --port "$PORT" >> "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
disown

for _ in $(seq 1 40); do
    [[ -f "$PID_FILE" ]] && break
    # Died during startup (bad config, port taken after all, import error...) - stop waiting.
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.25
done

if [[ ! -f "$PID_FILE" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "matricxon did not finish starting within 10s (pid $SERVER_PID still running) - check $LOG_FILE"
    else
        echo "matricxon exited during startup - last lines of $LOG_FILE:"
        tail -n 5 "$LOG_FILE"
    fi
    exit 1
fi

echo "matricxon started (pid $(cat "$PID_FILE")) on $HOST:$PORT, logging to $LOG_FILE"
