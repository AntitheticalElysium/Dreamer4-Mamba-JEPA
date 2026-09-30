#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
export JAX_PLATFORMS=cpu
grep -q 'LANE_MAMBA_LONG_POOLS_DONE' "$LOGDIR/lanes.log" || { echo 'long pools incomplete'; exit 2; }
[ -f artifacts/experiments/20260929_mamba_integration/long_resource_steady.json ] || { echo 'resource report absent'; exit 2; }
A=artifacts/eda/levers_mamba_long_worlds_v1
for spec in ldad1:full ldad1:reset6 raw:full ldad10:full; do
  arm=${spec%%:*}; mode=${spec##*:}
  name=long_corrg_${arm}_${mode}_b16_u3600_s7
  [ -f "$A/${name}.pt" ] || job mamba_long_${arm}_${mode} 4800 "$PY" artifacts/experiments/20260929_mamba_integration/long_train.py --arm "$arm" --mode "$mode" --batch 16 --total 3600
  [ -f "$A/${name}.pt" ] || { echo "missing completed long world: $name"; exit 2; }
done
echo "$(date '+%F %T') LANE_MAMBA_LONG_TRAIN_DONE" >> "$LOGDIR/lanes.log"
