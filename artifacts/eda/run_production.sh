#!/usr/bin/env bash
# The repaired Direct stack through the real production objective.
# Stage 1 rebuilds the latent cache at the repaired 64x16 geometry (resumable mmap
# store, ~13 GB); stage 2 trains 20k production dynamics steps, which exercise the
# two-step generated-prefix path the diagnostic fork objective never touched.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
echo "=== P1_production_1b  $(date -Is) ===" | tee -a triage.log
s=$SECONDS
if "$PY" run_production_1b.py --steps 20000 >>logs/P1_production_1b.log 2>&1; then
  echo "    ok    $((SECONDS-s))s" | tee -a triage.log
else
  echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log
  tail -30 logs/P1_production_1b.log | tee -a triage.log
fi
echo "production 1b finished $(date -Is)" >> triage.log
