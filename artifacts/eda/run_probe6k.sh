#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
for m in 500 1500 3000 4500 6000; do
  for a in 32x16 64x16; do
    $PY probe_capacity_arms.py --milestone $m --arm $a >>logs/82_probe6k.log 2>&1
  done
done
echo "probe 6k finished $(date -Is)" >> triage.log
