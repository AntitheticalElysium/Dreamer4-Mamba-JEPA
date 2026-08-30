#!/usr/bin/env bash
# Factual then counterfactual, sequentially. ~3.1h each at the measured 0.558 s/step.
# Not resumable with optimizer state: an interrupted arm restarts from zero.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
for arm in factual counterfactual; do
  [ -f "terminal_${arm}/training_report.json" ] && continue
  echo "=== U1_train_${arm}  $(date -Is) ===" | tee -a triage.log
  s=$SECONDS
  if "$PY" train_terminal_arms.py --arm "$arm" >>"logs/U1_train_${arm}.log" 2>&1; then
    echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else
    echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log
    tail -20 "logs/U1_train_${arm}.log" | tee -a triage.log
  fi
done

# death transfer at the first complete pass and at the end, then the regression gates
for arm in factual counterfactual; do
  [ -f "terminal_${arm}/world.pt" ] || continue
  for m in 13592 0; do
    echo "=== U2_death_${arm}_${m}  $(date -Is) ===" | tee -a triage.log
    "$PY" evaluate_death_transfer.py --arm "$arm" --milestone "$m" \
      >>"logs/U2_death_${arm}_${m}.log" 2>&1 && echo "    ok" | tee -a triage.log \
      || { echo "    FAIL" | tee -a triage.log; tail -12 "logs/U2_death_${arm}_${m}.log" | tee -a triage.log; }
  done
  echo "=== U3_gates_${arm}  $(date -Is) ===" | tee -a triage.log
  "$PY" evaluate_production_1b.py --world "terminal_${arm}/world.pt" --tag "$arm" \
    >>"logs/U3_gates_${arm}.log" 2>&1 && echo "    ok" | tee -a triage.log \
    || { echo "    FAIL" | tee -a triage.log; tail -12 "logs/U3_gates_${arm}.log" | tee -a triage.log; }
done
echo "terminal A/B finished $(date -Is)" >> triage.log
