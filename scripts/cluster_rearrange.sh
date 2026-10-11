#!/usr/bin/env bash
# Sourced by cluster_doctor.sh for --force: rebuilds the cluster from the nodes that answer (see
# app/cluster/rearrange.py). The best node keeps the cluster state and becomes the only voter, every other node
# is reset to fresh (its own journal and saved chat caches go; models and shared folders are never touched),
# joins as a learner, and learners are made voters until there are three. Ends with a short report.

wait_for() {  # wait_for "what" TRIES command...: polls every 2 s
  local what=$1 tries=$2 i; shift 2
  for ((i = 0; i < tries; i++)); do "$@" && return 0; sleep 2; done
  echo "  timed out waiting for: $what" >&2; return 1
}
field() {  # field URL PATH EXPR: a value from a node's JSON (python expression on d)
  curl -fsS -m 4 "$1$2" 2>/dev/null | "$PY" -c "import json,sys; d=json.load(sys.stdin); print($3)" 2>/dev/null
}
is_fresh() { [ "$(field "$1" /api/cluster/fresh 'd["fresh"]')" = True ]; }
runs_consensus() { [ "$(field "$1" /api/consensus/status 'd["enabled"]')" = True ]; }
sole_leader() { [ "$(field "$1" /api/consensus/status '(d["role"], len(d["voters"]))')" = "('leader', 1)" ]; }
voter_count() { field "$1" /api/consensus/status 'len(d["voters"])'; }
note() { REPORT+=("$1"); }

rearrange() {
  local plan kind a b c i
  probe "$WORK/a"; discover "$WORK/a"
  plan="$("$PY" -m app.cluster.rearrange "$WORK/a")" || { echo "${R}Cannot rearrange:${N} see above" >&2; return 1; }
  local DROPS=() OURL=() OADV=() DOWNS=() QVIA="" SEED_URL="" SEED_ID="" SEED_ADDR="" MODELS="" CACHE="" REPORT=()
  while IFS=$'\t' read -r kind a b c; do
    case "$kind" in
      seed) SEED_URL=$a SEED_ID=$b SEED_ADDR=$c ;; quorum) QVIA=$a ;; paths) MODELS=$a CACHE=$b ;;
      drop) DROPS+=("$a") ;; reset) OURL+=("$a") ;; join) OADV+=("$b") ;; down) DOWNS+=("$a") ;;
    esac
  done <<< "$plan"
  local want=$(( ${#OURL[@]} + 1 )); [ "$want" -gt 3 ] && want=3
  echo "Rearranging the cluster:"
  echo "  keeps its state:  $SEED_URL (becomes the only voter, then the leader)"
  echo "  reset to fresh:   ${OURL[*]:-nobody}   (their own journals and saved chat caches are deleted)"
  echo "  not answering:    ${DOWNS[*]:-nobody}   (skipped: bring them back, reset them, add them from pAIring)"
  echo "  goal:             $want voter(s), the rest ordinary nodes"
  typed "Chats are interrupted for a moment while this runs." rearrange || { echo "Cancelled."; return 1; }

  if [ ${#DROPS[@]} -gt 0 ]; then  # 1. the seed alone
    if [ "$QVIA" != none ]; then
      for i in "${DROPS[@]}"; do curl -fsS -m 30 -X DELETE "$QVIA/api/consensus/voters/$i?leave=true" > /dev/null || true; done
    elif [ "$(local_id)" = "$SEED_ID" ]; then
      scripts/stop.sh; "$PY" -m app.consensus.recover --force-new-group --confirm "$SEED_ID"; scripts/start.sh
      note "The old journal of $SEED_URL is backed up as data/consensus.journal.bak-* (the other voters' newest writes are gone)."
    else
      echo "${R}No majority: run this on the machine of $SEED_URL (it forces the group of one there).${N}" >&2; return 1
    fi
    wait_for "$SEED_URL to lead alone" 30 sole_leader "$SEED_URL" || return 1
  fi
  local failed=()
  for i in "${!OURL[@]}"; do  # 2. every other node from scratch
    local url=${OURL[$i]} code
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST "$url/api/cluster/reset")
    [ "$code" = 409 ] && code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST "$url/api/cluster/reset?force=true")
    [ "$code" = 200 ] && sleep 3 && wait_for "$url to be fresh" 60 is_fresh "$url" || { failed+=("$url"); continue; }
    local body; body=$("$PY" -c 'import json,sys; print(json.dumps({"models_dir": sys.argv[1], "prompt_cache_dir": sys.argv[2], "restart": True, "advertise_url": sys.argv[3], "consensus_seeds": {sys.argv[4]: sys.argv[5]}}))' "$MODELS" "$CACHE" "${OADV[$i]}" "$SEED_ID" "$SEED_ADDR")
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST -H 'Content-Type: application/json' -d "$body" "$url/api/cluster/provision")
    [ "$code" = 200 ] && sleep 3 && wait_for "$url to join" 60 runs_consensus "$url" || failed+=("$url")
  done
  for i in $(seq 1 40); do  # 3. voters up to the goal (the leader does it by itself too, this just does not wait)
    [ "$(voter_count "$SEED_URL")" -ge "$want" ] 2>/dev/null && break
    probe "$WORK/a"; analyze "$WORK/a"
    action=$(awk -F'\t' '$5 ~ /^add-voter/ {print $5; exit}' "$WORK/findings")
    [ -n "$action" ] && YES=1 apply "$action" > /dev/null
    sleep 3
  done
  probe "$WORK/a"; analyze "$WORK/a"
  report "$SEED_URL" "$want" "${#OURL[@]}" "${failed[*]:-}" "${DOWNS[*]:-}"
}

report() {  # report SEED WANT NODES FAILED DOWN
  local file="logs/cluster_rearrange_$(date +%Y%m%d-%H%M%S).txt"; mkdir -p logs
  {
    echo "Cluster rearrangement report"
    echo "  cluster id: $(field "$1" /api/consensus/status 'd["cluster_id"]')"
    echo "  leader:     $(field "$1" /api/consensus/status 'd["leader"]') voters: $(voter_count "$1") of goal $2, nodes: $(field "$1" /api/consensus/status 'len(d["nodes"])')"
    [ -n "$4" ] && echo "  NOT rearranged (reset or join failed): $4"
    [ -n "$5" ] && echo "  skipped (did not answer): $5"
    echo "  notes:"
    echo "   - $1 kept the cluster state (config, leases, cluster id); the other nodes were reset to fresh."
    echo "   - Models and the shared folders were not touched; reset nodes lost their own journals and saved chat caches."
    echo "   - pAIring follows the cluster's node list within about 15 seconds; its host list needs no change."
    echo "   - A skipped node must be reset before it is added again (pAIring > Add nodes > Reset to fresh)."
    for line in "${REPORT[@]:-}"; do [ -n "$line" ] && echo "   - $line"; done
    echo "  state now:"; show | sed 's/^/    /'
  } | tee "$file"
  echo "(saved to $file)"
}
