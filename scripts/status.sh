#!/usr/bin/env bash
# Reports whether matricxon is running and whether it actually answers
# GET /api/tags (pAIring's own liveness probe - see the plan for why that
# endpoint specifically matters).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${MATRICXON_HOST:-127.0.0.1}"
PORT="${MATRICXON_PORT:-8420}"
PID_FILE="run/matricxon.pid"

if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "matricxon: stopped"
    exit 1
fi

PID="$(cat "$PID_FILE")"

if curl -fsS -o /dev/null -m 3 "http://${HOST}:${PORT}/api/tags"; then
    echo "matricxon: running (pid $PID), healthy on http://${HOST}:${PORT}"
    exit 0
else
    echo "matricxon: running (pid $PID), but not answering on http://${HOST}:${PORT}"
    exit 2
fi
