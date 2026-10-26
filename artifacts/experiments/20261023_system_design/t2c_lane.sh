#!/usr/bin/env bash
# T2c lane: each line of $1 is "name anti H seed extra-args..."; train with t2c_joint.py, evaluate with t2_eval.py.
set -u
R=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
DATA=$R/artifacts/eda/system_design_v1/t2_data; OUT=$R/artifacts/eda/system_design_v1/t2c_runs
PY=$R/.venv/bin/python; X=$R/artifacts/experiments/20261023_system_design
while read -r name anti H seed extra; do
  run=$OUT/${name}_s${seed}
  if [ -f $run/eval.json ]; then echo "skip $run"; continue; fi
  if [ ! -f $run/done.json ]; then $PY $X/t2c_joint.py --data $DATA --out $run --anti $anti --H $H --seed $seed $extra || { echo "TRAIN FAILED $run"; continue; }; fi
  PYTHONPATH=$X $PY $X/t2_eval.py --data $DATA --run $run || echo "EVAL FAILED $run"
done < "$1"
echo "lane $1 complete"
