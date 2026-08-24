#!/usr/bin/env bash
# Round 3: all-action positive control, mini-H2, and the frozen-arm plateau test.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
LAST_OK=1
stage() {
  local name="$1"; shift
  echo "=== $name  $(date -Is) ===" | tee -a triage.log
  local s=$SECONDS
  if "$@" >"logs/$name.log" 2>&1; then
    echo "    ok    $((SECONDS-s))s" | tee -a triage.log; LAST_OK=1
  else
    echo "    FAIL  $((SECONDS-s))s  (logs/$name.log)" | tee -a triage.log
    tail -25 "logs/$name.log" | tee -a triage.log; LAST_OK=0
  fi
}
mkdir -p logs

# Test 2 -- all-action positive control (~35 min)
stage 12_train_allaction "$PY" train_allaction.py --steps 20000
for m in 005000 010000 020000; do
  [ -f "damage_allaction/model_${m}.pt" ] && stage "13_eval_allaction_${m}" \
    "$PY" evaluate_damage_classifier.py --model "damage_allaction/model_${m}.pt" \
    --out damage_allaction --test-roots-only
done
# the factual arm scored on the identical test roots, for a like-for-like number
stage 14_eval_factual_testroots "$PY" evaluate_damage_classifier.py \
  --model damage_classifier/model_020000.pt --out damage_classifier --test-roots-only

# Test 3 -- mini-H2: Phase-1A encoder, unfrozen (~3 h)
stage 15_miniH2_preflight "$PY" train_damage_pixels.py --preflight --init-phase1a \
  --out damage_miniH2
if [ "$LAST_OK" = "1" ]; then
  stage 16_train_miniH2 "$PY" train_damage_pixels.py --steps 20000 --init-phase1a \
    --out damage_miniH2
fi
for m in 005000 010000 020000; do
  [ -f "damage_miniH2/model_${m}.pt" ] && stage "17_eval_miniH2_${m}" \
    "$PY" evaluate_damage_pixels.py --model "damage_miniH2/model_${m}.pt" \
    --out damage_miniH2
done

# Plateau test -- frozen arm to 80k (~2.2 h)
stage 18_train_frozen_80k "$PY" train_damage_classifier.py --steps 80000 \
  --milestones 20000 40000 80000 --out damage_frozen80k
for m in 020000 040000 080000; do
  [ -f "damage_frozen80k/model_${m}.pt" ] && stage "19_eval_frozen80k_${m}" \
    "$PY" evaluate_damage_classifier.py --model "damage_frozen80k/model_${m}.pt" \
    --out damage_frozen80k
done
echo "round 3 finished $(date -Is)" >> triage.log
