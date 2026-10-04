#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
A=artifacts/eda/levers_mamba_integration_v1
for pool in raw ldad1 ldad10; do
  name=int_corrg_${pool}_suffix_s7_fmamba_u18000
  [ -f "$A/${name}_at6000.pt" ] || { echo "missing immutable 6k checkpoint: $name"; exit 2; }
  [ -f "$A/${name}_at12000.pt" ] || job mamba_${pool}_short12k 3600 "$PY" artifacts/experiments/20260929_mamba_integration/short_train.py --pool "$pool" --until 12000
  [ -f "$A/${name}_at12000.pt" ] || { echo "missing immutable 12k checkpoint: $name"; exit 2; }
done
echo "$(date '+%F %T') LANE_MAMBA_SHORT12K_DONE" >> "$LOGDIR/lanes.log"
