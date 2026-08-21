#!/usr/bin/env bash
# Mini-H2, queued behind the plateau test now that its preflight passes.
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
while ! grep -q "round 3 finished" triage.log; do sleep 60; done
stage 20_train_miniH2 "$PY" train_damage_pixels.py --steps 20000 --init-phase1a \
  --out damage_miniH2
for m in 005000 010000 020000; do
  [ -f "damage_miniH2/model_${m}.pt" ] && stage "21_eval_miniH2_${m}" \
    "$PY" evaluate_damage_pixels.py --model "damage_miniH2/model_${m}.pt" --out damage_miniH2
done
echo "round 4 finished $(date -Is)" >> triage.log
