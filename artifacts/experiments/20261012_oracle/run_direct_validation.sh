#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_AUTOTUNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/artifacts/eda/oracle_v1/direct_generator/runtime/triton_cache"
DIRECT_SCRIPT=artifacts/experiments/20261012_oracle
DIRECT_OUT=artifacts/eda/oracle_v1/direct_generator
mkdir -p artifacts/eda/levers_logs
exec >> artifacts/eda/levers_logs/direct_generator_validation_20261014.log 2>&1
.venv/bin/python "$DIRECT_SCRIPT/direct_train.py" --updates 3000 --batch 16 --directory "$DIRECT_OUT/smoke" --stop-after 2
.venv/bin/python "$DIRECT_SCRIPT/direct_train.py" --updates 3000 --batch 16 --directory "$DIRECT_OUT/smoke" --stop-after 4
.venv/bin/python "$DIRECT_SCRIPT/direct_train.py" --updates 3000 --batch 16 --directory "$DIRECT_OUT/smoke_reference" --stop-after 4
.venv/bin/python "$DIRECT_SCRIPT/direct_checks.py" --resume-proof
