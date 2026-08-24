#!/usr/bin/env bash
# Paired-data scaling curve. Waits for mini-H2 and the rescore, then runs four nested
# rungs spanning an 8x range of unique hazard-choice roots at a fixed 20k budget.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
stage() {
  local name="$1"; shift
  echo "=== $name  $(date -Is) ===" | tee -a triage.log
  local s=$SECONDS
  if "$@" >"logs/$name.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s  (logs/$name.log)" | tee -a triage.log
       tail -25 "logs/$name.log" | tee -a triage.log; fi
}
mkdir -p logs
while [ ! -f branched_damage/manifest.json ]; do sleep 60; done
while ! grep -q "round 4 finished" triage.log; do sleep 60; done

# the largest rung is whatever the rescore actually yielded; the rest are nested halves
MAX=$("$PY" - <<'EOF'
import sys; sys.path.insert(0, ".")
from train_paired_scaling import load_pool
roots, _ = load_pool()
print(sum(1 for r in roots if r["split"] == "fit" and r["hazard"] and not r["reserved"]))
EOF
)
echo "available fit hazard-choice roots: $MAX" | tee -a triage.log
R4=$MAX; R3=$((MAX / 2)); R2=$((MAX / 4)); R1=$((MAX / 8))
echo "rungs: $R1 $R2 $R3 $R4" | tee -a triage.log

stage 23_scaling_preflight "$PY" train_paired_scaling.py --roots "$R1" \
  --out scaling/preflight --preflight
for r in "$R1" "$R2" "$R3" "$R4"; do
  stage "24_scaling_${r}" "$PY" train_paired_scaling.py --roots "$r" \
    --steps 20000 --out "scaling/k${r}"
  [ -f "scaling/k${r}/model.pt" ] && stage "25_eval_scaling_${r}" \
    "$PY" evaluate_paired.py --model "scaling/k${r}/model.pt" --out "scaling/k${r}"
done
echo "round 5 finished $(date -Is)" >> triage.log
