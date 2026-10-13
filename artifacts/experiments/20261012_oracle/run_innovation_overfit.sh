#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_AUTOTUNING=1
INNOVATION_DRIVER=$(rg -o '[0-9]{3}\.[0-9]{2}\.[0-9]{2}' /proc/driver/nvidia/version | head -n 1)
INNOVATION_RUNTIME="$PWD/artifacts/eda/oracle_v1/generated_prefix/runtime_${INNOVATION_DRIVER//./_}"
export TRITON_CACHE_DIR="$INNOVATION_RUNTIME/triton_cache"
if ! nvidia-smi --query-gpu=driver_version --format=csv,noheader >/dev/null 2>&1; then
    [[ -f "$INNOVATION_RUNTIME/usr/lib/libcuda.so.1" && -f "$INNOVATION_RUNTIME/usr/lib/libnvidia-ml.so.1" ]]
    export LD_LIBRARY_PATH="$INNOVATION_RUNTIME/usr/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
{
    date '+START %Y-%m-%d %H:%M:%S %Z'
    while systemctl --user is-active --quiet learned_innovation_eval_20261013_v2.service; do sleep 10; done
    .venv/bin/python artifacts/experiments/20261012_oracle/innovation_overfit.py --stop-after 500
    CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import sys,torch
from pathlib import Path
sys.path.insert(0,'artifacts/experiments/20261012_oracle')
import loss_intervention as L
d=L.OUT/'learned_innovation/optimization_control'
s=torch.load(d/'state.pt',map_location='cpu',weights_only=False)
assert s['update']==500
L.atomic({'contract_id':s['contract_id'],'update':s['update'],'history':s['history'],
          'scope':'initial500-update overfit review; supplied-code positive control required for interpreting encoder failure'},
         d/'review_u0500.json',True)
PY
    date '+REVIEW READY %Y-%m-%d %H:%M:%S %Z'
} >> artifacts/eda/levers_logs/learned_innovation_overfit.log 2>&1
