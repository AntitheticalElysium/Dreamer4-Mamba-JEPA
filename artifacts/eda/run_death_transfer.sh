#!/usr/bin/env bash
# Corrected death transfer on both arms, identical seed_split held-out roots.
# abm0 trained on all 17 successors at each root; production saw only ordinary
# trajectories. If abm0 clears the action-only floor and production does not, the
# architecture can learn state-conditioned death and production lacks the exposure.
# If neither clears it, all-17 supervision did not solve termination either.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
for arm in production abm0 abm1; do
  echo "=== R1_death_${arm}  $(date -Is) ===" | tee -a triage.log
  if "$PY" evaluate_death_transfer.py --arm "$arm" >>"logs/R1_death_${arm}.log" 2>&1; then
    echo "    ok" | tee -a triage.log
  else
    echo "    FAIL" | tee -a triage.log; tail -12 "logs/R1_death_${arm}.log" | tee -a triage.log
  fi
done
echo "death transfer arms finished $(date -Is)" >> triage.log
