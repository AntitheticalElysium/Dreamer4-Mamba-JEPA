#!/usr/bin/env bash
# Resumable EDA queue.
#
# Every stage is idempotent: a finished stage is skipped on relaunch, and each
# training job restores from its own `resume.pt` (modules, optimizer, step counter,
# both RNG streams). The queue can be killed at any moment -- Ctrl-C, kill, reboot --
# and restarted with exactly this command, losing at most `--resume-every` steps.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu

stage() {
  local name="$1"; shift
  echo "=== $name  $(date -Is) ===" | tee -a triage.log
  local s=$SECONDS
  if "$@" >>"logs/$name.log" 2>&1; then
    echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else
    echo "    FAIL  $((SECONDS-s))s  (logs/$name.log)" | tee -a triage.log
    tail -25 "logs/$name.log" | tee -a triage.log
  fi
}
mkdir -p logs

# --- mini-H2: the Phase-1A encoder, unfrozen ---------------------------------
if [ ! -f damage_miniH2/training_report.json ]; then
  stage 30_train_miniH2 "$PY" train_damage_pixels.py --steps 20000 --init-phase1a \
    --out damage_miniH2
fi
for m in 005000 010000 020000; do
  if [ -f "damage_miniH2/model_${m}.pt" ] && \
     [ ! -f "damage_miniH2/evaluation_model_${m}.json" ]; then
    stage "31_eval_miniH2_${m}" "$PY" evaluate_damage_pixels.py \
      --model "damage_miniH2/model_${m}.pt" --out damage_miniH2
  fi
done

# --- paired-data scaling curve, nested rungs over an 8x range ----------------
MAX=$("$PY" - <<'EOF'
import sys
sys.path.insert(0, ".")
from train_paired_scaling import load_pool

roots, _ = load_pool()
print(sum(1 for r in roots if r["split"] == "fit" and r["hazard"] and not r["reserved"]))
EOF
)
R4=$MAX; R3=$((MAX / 2)); R2=$((MAX / 4)); R1=$((MAX / 8))
echo "scaling rungs from $MAX fit hazard roots: $R1 $R2 $R3 $R4" | tee -a triage.log
for r in "$R1" "$R2" "$R3" "$R4"; do
  if [ ! -f "scaling/k${r}/training_report.json" ]; then
    stage "32_scaling_${r}" "$PY" train_paired_scaling.py --roots "$r" --steps 20000 \
      --out "scaling/k${r}"
  fi
  if [ -f "scaling/k${r}/model.pt" ] && [ ! -f "scaling/k${r}/evaluation.json" ]; then
    stage "33_eval_scaling_${r}" "$PY" evaluate_paired.py \
      --model "scaling/k${r}/model.pt" --out "scaling/k${r}"
  fi
done

echo "eda queue finished $(date -Is)" >> triage.log
