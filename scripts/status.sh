#!/usr/bin/env bash
# Reports whether matricxon is running and whether it actually answers
# GET /api/tags (pAIring's own liveness probe - see the plan for why that
# endpoint specifically matters).
set -euo pipefail

# Colored like uvicorn's own "INFO:" lines (plain text when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

HOST="${MATRICXON_HOST:-127.0.0.1}"
PORT="${MATRICXON_PORT:-8420}"
PID_FILE="run/matricxon.pid"

if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    err "matricxon: stopped"
    exit 1
fi

PID="$(cat "$PID_FILE")"

if curl -fsS -o /dev/null -m 3 "http://${HOST}:${PORT}/api/tags"; then
    info "matricxon: running (pid $PID), healthy on http://${HOST}:${PORT}"
    exit 0
else
    err "matricxon: running (pid $PID), but not answering on http://${HOST}:${PORT}"
    exit 2
fi
