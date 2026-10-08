#!/usr/bin/env bash
# Authorized E20 queue. No automatic NOTEBOOK writes or threshold/recipe changes.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
source artifacts/experiments/20260927_levers/lib.sh
export JAX_PLATFORMS=cpu
# All commands self-resume from committed source/input-bound states.
job e20_source_labels 0 "$PY" -B "$L/e20_data.py"
job e20_prepare 3800 "$PY" -B "$L/e20_prepare.py"
job e20_reader 0 "$PY" -B "$L/e20_verify.py" reader
job e20_mechanics 3400 "$PY" -B "$L/e20_verify.py" mechanics
W=artifacts/eda/levers_tworlds_v1
for seed in 7 8; do
  for arm in A B C; do
    if [ "$arm" = C ]; then
      "$PY" -B -c 'import json; assert json.load(open("artifacts/eda/levers_e20_v1/health_reader.json"))["passed"], "C held: factual health reader failed its predeclared validity check"'
    fi
    job "e20_${arm}_s${seed}_train" 3400 "$PY" -B "$L/e20_train.py" --arm "$arm" --seed "$seed"
  done
  worlds=()
  for arm in A B C; do worlds+=("$W/e20_${arm}_s${seed}_fmamba_fromM16.pt"); done
  job "e20_s${seed}_health_strict" 3000 "$PY" -B "$L/e19_eval.py" "${worlds[@]}"
  job "e20_s${seed}_health_w4" 3000 "$PY" -B "$L/check_damage.py" "${worlds[@]}" --window 4
  job "e20_s${seed}_health_w5" 3000 "$PY" -B "$L/check_damage.py" "${worlds[@]}" --window 5
  job "e20_s${seed}_health_w15" 3000 "$PY" -B "$L/check_damage.py" "${worlds[@]}" --window 15
  job "e20_s${seed}_world_w5" 3000 "$PY" -B artifacts/experiments/20261005_recovery/teval_export.py "${worlds[@]}"
  job "e20_s${seed}_world_w15" 3000 "$PY" -B artifacts/experiments/20261005_recovery/teval_export.py "${worlds[@]}" --window 15
  job "e20_s${seed}_recall" 3000 "$PY" -B "$L/check_recall.py" --futures "${worlds[@]}"
done
echo "$(date '+%F %T') E20_DONE" >> "$LOGDIR/lanes.log"
