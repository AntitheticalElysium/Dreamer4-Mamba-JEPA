#!/usr/bin/env bash
# T1 lane: train + evaluate each "arch seed" line of the list file $1. Do not edit while running.
set -u
R=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
DATA=$R/artifacts/eda/system_design_v1/t1_data; OUT=$R/artifacts/eda/system_design_v1/t1_runs
PY=$R/.venv/bin/python; X=$R/artifacts/experiments/20261023_system_design
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r arch seed; do
  run=$OUT/${arch}_s${seed}
  if [ -f $run/eval.json ]; then echo "skip $run"; continue; fi
  if [ ! -f $run/done.json ]; then $PY $X/t1_train.py --data $DATA --out $run --arch $arch --seed $seed || { echo "TRAIN FAILED $run"; continue; }; fi
  $PY $X/t1_eval.py --data $DATA --run $run || echo "EVAL FAILED $run"
done < "$1"
echo "lane $1 complete"
