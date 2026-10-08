#!/usr/bin/env bash
# Oct6 bounded frozen diagnostics. No new world training or recipe changes.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
W=artifacts/eda/levers_tworlds_v1

wait_set() {
  local seed=$1 bb=$2
  while [ ! -f "$W/state/e19_C_s${seed}_${bb}_from36000/complete.json" ] || \
        [ ! -f "$W/state/e19_B_s${seed}_${bb}_from36000/complete.json" ]; do
    local active
    active=$(systemctl --user show d4mj-oct05-lead.service -p ActiveState --value)
    if [ "$active" != active ] && [ "$active" != activating ]; then
      echo "Main queue stopped before e19_${bb}_s${seed}; preserve partial diagnosis and stop." >&2
      exit 1
    fi
    sleep 15
  done
}

# Already-saved 12k snapshots: descriptive learning-curve check, never the 36k endpoint verdict.
job e18_recall_s7_at12000 2000 $PY -B "$L/check_recall.py" \
  "$W/corrt_raw_teacher_s7_fcanvas_u36000_at12000.pt" \
  "$W/corrt_raw_teacher_s7_fmamba_u36000_at12000.pt"

for pair in '8 fmamba' '7 full' '8 full'; do
  read -r seed bb <<< "$pair"
  wait_set "$seed" "$bb"
  tag=${bb}_s${seed}
  job e19_router_$tag 1600 $PY -B "$L/e19_diagnose.py" \
    "$W/e19_C_s${seed}_${bb}_from36000.pt" "$W/e19_B_s${seed}_${bb}_from36000.pt"
  job e19_local_gradients_$tag 1600 $PY -B "$L/e19_gradients.py" --seed "$seed" --backbone "$bb"
  job e19_train_mechanism_$tag 1600 $PY -B "$L/e19_train_diagnose.py" --seed "$seed" --backbone "$bb"
  job e19_gradient_budget_$tag 0 $PY -B "$L/e19_gradient_budget.py" --seed "$seed" --backbone "$bb"
  job e19_trace_readout_$tag 0 $PY -B "$L/e19_trace_readout.py" --seed "$seed" --backbone "$bb"
done
echo "$(date '+%F %T') LANE86_DONE" >> "$LOGDIR/lanes.log"
