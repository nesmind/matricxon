#!/usr/bin/env bash
# Finds what is wrong with a cluster and, with --fix, repairs what can be repaired safely.
# Usage: scripts/cluster_doctor.sh [--fix | --force] [--yes] URL [URL...]      e.g. http://10.0.0.5:8420
# One node that answers is enough: the rest are found from the cluster's own node list (at the addresses the
# cluster uses, which this machine may not reach: those show as down). More URLs may be given.
# Without --fix it only reports (exit 0 healthy, 1 problems, 2 usage). Nothing destructive
# runs without a typed confirmation (--yes skips it). Models and shared folders are never touched.
#
# What it finds (app/cluster/doctor.py): nodes down, no majority of voters (-> forces a group of one on the
# best survivor: run --fix ON that machine), all voters lost, voters that no longer exist, voters whose
# address moved, learners still replaying the log or stuck (-> reset), learners never made voters,
# nodes taken out but still running, two different clusters, failing shared storage, nodes outside the
# cluster. Locally it also checks free disk space and the journal size.
#
# --force rebuilds the whole cluster instead (scripts/cluster_rearrange.sh): the best node keeps the state,
# every other node is reset and re-joined, up to three voters, then a short report (also saved in logs/).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-.venv/bin/python}"; [ -x "$PY" ] || PY=python3
FIX=0 YES=0 FORCE=0 URLS=()
for a in "$@"; do
  case "$a" in --fix) FIX=1 ;; --yes) YES=1 ;; --force) FORCE=1 ;; -h|--help) sed -n 2,17p "$0"; exit 0 ;; -*) echo "unknown option $a" >&2; exit 2 ;; *) URLS+=("${a%/}") ;; esac
done
[ ${#URLS[@]} -gt 0 ] || { echo "Give at least one node URL (see --help)." >&2; exit 2; }
if [ -t 1 ]; then R=$'\033[31m' Y=$'\033[33m' G=$'\033[32m' N=$'\033[0m'; else R="" Y="" G="" N=""; fi
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

probe() {  # probe DIR: what every node says, one set of files per node
  mkdir -p "$1"; local i=0 u
  for u in "${URLS[@]}"; do
    echo "$u" > "$1/$i.url"
    curl -fsS -m 4 "$u/api/consensus/status" -o "$1/$i.status" 2>/dev/null || rm -f "$1/$i.status"
    curl -fsS -m 4 "$u/api/cluster/readiness" -o "$1/$i.ready" 2>/dev/null || rm -f "$1/$i.ready"
    i=$((i + 1))
  done
}
discover() {  # discover DIR: add the cluster's other nodes, as the cluster knows them, to URLS (and probe them)
  local round new u
  for round in 1 2 3; do
    new=$("$PY" -m app.cluster.doctor_cli "$1" --discover 2>/dev/null)
    [ -z "$new" ] && return 0
    for u in $new; do URLS+=("$u"); done
    probe "$1"
  done
}
analyze() { "$PY" -m app.cluster.doctor_cli "$@" > "$WORK/findings" 2>"$WORK/err"; return $?; }
show() {
  local sev code node text action color
  while IFS=$'\t' read -r sev code node text action; do
    case "$sev" in error) color=$R ;; warn) color=$Y ;; *) color=$N ;; esac
    printf '%s[%s] %s%s  %s: %s\n' "$color" "$sev" "$code" "$N" "$node" "$text"
    [ -n "$action" ] && printf '        fix: %s\n' "$action"
  done < "$WORK/findings"
}
node_id_of() { "$PY" - "$WORK/a" "$1" <<'E'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
for u in d.glob("*.url"):
    if u.read_text().strip() == sys.argv[2]:
        for kind in ("status", "ready"):
            try: print(json.loads((d / f"{u.stem}.{kind}").read_text()).get("node_id") or "", end=""); raise SystemExit
            except (OSError, ValueError): pass
E
}
local_id() { cat data/node_id 2>/dev/null || true; }
local_checks() {
  [ -d data ] || return 0
  local free; free=$(df -P data | awk 'NR==2 {gsub("%","",$5); print 100-$5}')
  [ "${free:-100}" -lt 5 ] && echo "${R}[error] DISK_FULL${N}  this machine: only ${free}% free where data/ lives (the journal cannot be written)"
  local j; j=$(du -m data/consensus.journal 2>/dev/null | cut -f1)
  [ "${j:-0}" -gt 200 ] && echo "${Y}[warn] BIG_JOURNAL${N}  this machine: consensus.journal is ${j} MB (compaction only runs on a current Matricxon)"
  true
}
confirm() { [ "$YES" = 1 ] && return 0; read -r -p "$1 [y/N] " a; [ "$a" = y ]; }
typed() { [ "$YES" = 1 ] && return 0; read -r -p "$1 Type '$2' to continue: " a; [ "$a" = "$2" ]; }

apply() {  # apply "action line"
  set -- $1
  case "$1" in
    drop-voter) confirm "Take voter $3 out of the cluster through $2?" || return 0
      curl -fsS -m 30 -X DELETE "$2/api/consensus/voters/$3?leave=true" ;;
    add-voter) confirm "Make $3 a voter (address $4) through $2?" || return 0
      curl -fsS -m 30 -X POST -H 'Content-Type: application/json' -d "{\"id\":\"$3\",\"address\":\"$4\"}" "$2/api/consensus/voters" ;;
    recover)
      if [ "$(local_id)" != "$3" ]; then echo "  -> run on the machine of $2:  scripts/cluster_doctor.sh --fix ${URLS[*]}"; return 0; fi
      typed "This stops THIS node and makes it the only voter; the other voters are lost for good (journal backed up)." recover || return 0
      scripts/stop.sh; "$PY" -m app.consensus.recover --force-new-group --confirm "$3"; scripts/start.sh
      echo "  -> afterwards reinstall the other nodes FRESH (scripts/reset_cluster_node.sh) and add them again." ;;
    reset-node)
      local id; id=$(node_id_of "$2")
      if [ -z "$id" ] || [ "$(local_id)" != "$id" ]; then echo "  -> run on the machine of $2:  scripts/reset_cluster_node.sh --stop --yes && scripts/start.sh"; return 0; fi
      typed "This resets THIS node to a fresh install (models and shared folders are kept)." reset || return 0
      scripts/reset_cluster_node.sh --stop --yes && scripts/start.sh ;;
  esac
  echo
}

if [ "$FORCE" = 1 ]; then  # rebuild the whole cluster from the nodes that answer
  . "$(dirname "$0")/cluster_rearrange.sh"; rearrange; exit $?
fi
probe "$WORK/a"; discover "$WORK/a"
analyze "$WORK/a"; status=$?
if grep -q $'\tREPLAYING\t' "$WORK/findings"; then  # a learner may be slow or stuck: look again a moment later
  echo "A node is replaying the log: looking again in 10 seconds…"; sleep 10
  probe "$WORK/b"; analyze "$WORK/b" --prev "$WORK/a"; status=$?; rm -rf "$WORK/a"; mv "$WORK/b" "$WORK/a"
fi
[ -s "$WORK/err" ] && cat "$WORK/err" >&2
show; local_checks
[ -s "$WORK/findings" ] || echo "${G}OK${N}: ${#URLS[@]} node(s) checked, nothing wrong."
if [ "$FIX" = 1 ] && [ -s "$WORK/findings" ]; then
  echo; echo "Fixing…"
  while IFS=$'\t' read -r sev code node text action; do [ -n "$action" ] && { echo "$code:"; apply "$action"; }; done < "$WORK/findings"
  echo "Checking again…"; sleep 3; probe "$WORK/c"; analyze "$WORK/c"; status=$?; show
fi
exit $((status > 0 ? 1 : 0))
