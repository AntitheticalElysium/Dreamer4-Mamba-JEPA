#!/usr/bin/env bash
# Frozen CPU endpoint diagnosis; original world training stays unchanged.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e18_endpoint_s7_cpu 0 $PY -B "$L/e18_endpoint_cpu.py" --seed 7
job e19_exposure_census_cpu 0 $PY -B "$L/e19_exposure_census.py"
job e19_health_conditioning_cpu 0 $PY -B "$L/e19_health_conditioning.py"
W=artifacts/eda/levers_tworlds_v1
if [ -f "$L/e18_seed8_hold.json" ]; then
  echo "$(date '+%F %T') LANE90_DONE s7_complete_s8_parked_at24000" >> "$LOGDIR/lanes.log"
  exit 0
fi
while [ ! -f "$W/corrt_raw_teacher_s8_fcanvas_u36000.pt" ]; do
  active=$(systemctl --user show d4mj-oct05-lead.service -p ActiveState --value)
  if [ "$active" != active ] && [ "$active" != activating ]; then
    echo "Main queue stopped before canvas s8; preserve CPU evidence and stop." >&2
    exit 1
  fi
  sleep 20
done
job e18_endpoint_s8_cpu 0 $PY -B "$L/e18_endpoint_cpu.py" --seed 8
echo "$(date '+%F %T') LANE90_DONE" >> "$LOGDIR/lanes.log"
