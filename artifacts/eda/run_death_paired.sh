#!/usr/bin/env bash
# Re-score every checkpoint so paired differences can be bootstrapped over the same
# 197 test roots. The true-successor encoding is cached, so each pass is ~30s.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
"$PY" evaluate_death_transfer.py --arm production >>logs/V1_paired.log 2>&1
for arm in factual counterfactual; do
  for m in 5000 10000 13592 0; do
    "$PY" evaluate_death_transfer.py --arm "$arm" --milestone "$m" >>logs/V1_paired.log 2>&1
  done
done
echo "death paired rescore finished $(date -Is)" >> triage.log
