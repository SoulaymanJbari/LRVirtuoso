#!/bin/bash
# collect_stats.sh — copy each run's sim.stats into one flat folder, renamed by run.
#
# usage: ./collect_stats.sh DEST EXP_DIR [EXP_DIR ...]
#   ex : ./collect_stats.sh ~/nest_stats exp_k20-real exp_k20-latency
#
# exp_k20-real/results/k20-baseline_cc/simulation/sim.stats
#   -> DEST/k20-real__k20-baseline_cc.stats
#
# sim.stats is looked up in <run>/simulation/ first, then in <run>/ itself.
# Runs without sim.stats (still running or failed) are listed, not copied.

set -uo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 DEST EXP_DIR [EXP_DIR ...]" >&2
  exit 1
fi

DEST="$1"; shift
mkdir -p "$DEST"

copied=0; missing=0
for exp in "$@"; do
  exp="${exp%/}"
  if [ ! -d "$exp/results" ]; then
    echo "skip: $exp (no results/ folder)" >&2
    continue
  fi
  suite="$(basename "$exp")"
  suite="${suite#exp_}"                      # exp_k20-real -> k20-real
  for run in "$exp"/results/*/; do
    run="${run%/}"
    name="$(basename "$run")"
    stats=""
    for cand in "$run/simulation/sim.stats" "$run/sim.stats"; do
      if [ -f "$cand" ]; then stats="$cand"; break; fi
    done
    if [ -n "$stats" ]; then
      cp -p "$stats" "$DEST/${suite}__${name}.stats"
      copied=$((copied + 1))
    else
      echo "no sim.stats: $suite / $name" >&2
      missing=$((missing + 1))
    fi
  done
done

echo "$copied file(s) copied to $DEST, $missing run(s) without sim.stats"
