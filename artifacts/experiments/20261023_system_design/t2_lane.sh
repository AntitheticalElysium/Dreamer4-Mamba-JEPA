#!/usr/bin/env bash
# T2 lane: train + evaluate each config in the list file given as $1 (one "anti H seed" per line). Do not edit while running.
set -u
R=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
DATA=$R/artifacts/eda/system_design_v1/t2_data
OUT=$R/artifacts/eda/system_design_v1/t2_runs
PY=$R/.venv/bin/python
X=$R/artifacts/experiments/20261023_system_design
while read -r anti H seed; do
  run=$OUT/${anti}_H${H}_s${seed}
  if [ -f $run/eval.json ]; then echo "skip $run"; continue; fi
  if [ ! -f $run/done.json ]; then
    $PY $X/t2_joint.py --data $DATA --out $run --anti $anti --H $H --seed $seed || { echo "TRAIN FAILED $run"; continue; }
  fi
  $PY $X/t2_eval.py --data $DATA --run $run || echo "EVAL FAILED $run"
done < "$1"
echo "lane $1 complete"
