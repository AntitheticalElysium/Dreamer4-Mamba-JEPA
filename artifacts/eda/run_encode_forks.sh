#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
for n in 32 64; do
  [ -f "forkset_s1_n${n}/manifest.json" ] || \
    $PY encode_fork_dataset.py --n-latents $n --suffix s1 --milestone 6000 \
      --out "forkset_s1_n${n}" >>logs/94_encode_n${n}.log 2>&1
done
echo "fork encoding finished $(date -Is)" >> triage.log
