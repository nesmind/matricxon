#!/usr/bin/env bash
# Hard reset: makes THIS node a fresh install (leaves any cluster). Run on the node itself.
# Removes the cluster keys from .env, the consensus journal, the saved-cache policy and the
# node identity (a new one is made at start, so the old cluster no longer knows it), and
# this node's markers in the shared folder, and its own saved chat caches.
# Never touches models or anything shared. Stop the server first (or pass --stop).
# Usage: scripts/reset_cluster_node.sh [--yes] [--stop] [--env FILE] [--data DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=.env DATA=data YES=0 STOP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) YES=1 ;; --stop) STOP=1 ;;
    --env) ENV="$2"; shift ;; --data) DATA="$2"; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
# Only a process that IS the server (its command line starts with python... uvicorn app.main:app), never
# a shell or editor that merely mentions it.
SERVER='^[^ ]*python[^ ]* [^ ]*uvicorn app\.main:app'
if pgrep -f "$SERVER" >/dev/null; then
  if [ "$STOP" = 1 ]; then pkill -f "$SERVER"; sleep 2
  else echo "Matricxon is running: stop it first (or use --stop)." >&2; exit 1; fi
fi
echo "This will reset the node: cluster keys in $ENV, $DATA/consensus.journal*,"
echo "$DATA/prompt_cache_policy.json, $DATA/node_id and its own saved chat caches (not shared ones)."
echo "Models are kept."
if [ "$YES" != 1 ]; then read -r -p "Type 'reset' to continue: " a; [ "$a" = reset ] || exit 1; fi
# Cluster markers in the shared folder (not data): this node's own file and its seed claim.
CACHE="$(grep -E '^MATRICXON_PROMPT_CACHE_DIR=' "$ENV" 2>/dev/null | cut -d= -f2- || true)"
ID="$(cat "$DATA/node_id" 2>/dev/null || true)"
if [ -n "$CACHE" ] && [ -n "$ID" ]; then
  M="$(dirname "$CACHE")/.cluster"
  rm -f "$M/nodes/$ID.json"
  grep -q "\"$ID\"" "$M/seed.json" 2>/dev/null && rm -f "$M/seed.json"
fi
if [ -f "$ENV" ]; then
  sed -i -E '/^(MATRICXON_CLUSTER_ROLE|MATRICXON_MODELS_DIR|MATRICXON_PROMPT_CACHE_DIR|MATRICXON_ADVERTISE_URL|MATRICXON_CONSENSUS_SEEDS)=/d' "$ENV"
fi
rm -f "$DATA"/consensus.journal* "$DATA"/prompt_cache_policy.json "$DATA"/node_id
[ -z "$CACHE" ] && rm -f "$DATA"/prompt_cache/*.mxc  # only its own folder, never a shared one
echo "Done. Start Matricxon again: it is a fresh node."
