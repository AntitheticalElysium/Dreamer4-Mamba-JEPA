#!/usr/bin/env bash
# Inference follows training; readout labels never supervise sampled branches.
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
INNOVATION_RUN=artifacts/eda/oracle_v1/learned_innovation
{
    date '+START %Y-%m-%d %H:%M:%S %Z'
    while [[ ! -f "$INNOVATION_RUN/training.json" ]]; do
        if ! systemctl --user is-active --quiet learned_innovation_20261013.service; then
            printf 'FAILED: training stopped before its declared budget\n'
            exit 1
        fi
        sleep 10
    done
    while systemctl --user is-active --quiet learned_innovation_20261013.service; do sleep 10; done
    [[ -f "$INNOVATION_RUN/real_futures/fan.pt" ]]
    .venv/bin/python artifacts/experiments/20261012_oracle/innovation_eval_checks.py
    for update in 0 500 1500 3000; do
        .venv/bin/python artifacts/experiments/20261012_oracle/innovation_eval.py --updates "$update" --horizon 1 --samples 16 --batch 4
        CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/experiments/20261012_oracle/innovation_analysis.py --updates "$update" --horizon 1
    done
    .venv/bin/python artifacts/experiments/20261012_oracle/innovation_content.py export --updates 3000
    CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/experiments/20261012_oracle/innovation_content.py score --updates 3000
    # A constant current code cannot establish innovation completion. Do not
    # turn an unused channel into another expensive recursive comparison.
    INNOVATION_ACTIVE_ARMS=$(CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import sys,torch,json
from pathlib import Path
sys.path.insert(0,'artifacts/experiments/20261012_oracle')
import loss_intervention as L
p=L.OUT/'learned_innovation';directory=p/'eval_u3000_h01'
active=[];records={}
for arm in ('capacity','rate'):
    q=torch.load(directory/f'pred_{arm}_posterior.pt',map_location='cpu',weights_only=False)
    shuffled=torch.load(directory/f'pred_{arm}_shuffle_current.pt',map_location='cpu',weights_only=False)
    n=int(q['codes'][:,0,0].unique(dim=0).shape[0])
    effect=float((q['predictions']-shuffled['predictions']).abs().max())
    records[arm]={'distinct_joint_codes':n,'shuffle_output_maxabs':effect}
    if n>1 and effect>0:active.append(arm)
L.atomic({'current_code_use':records,'recursive_candidates':active,
          'scope':'necessary executed code-use check; not a capability pass or proof of learned aleatoric separation'},
         p/'recursive_queue_decision.json',True)
print(' '.join(active))
PY
)
    if [[ -n "$INNOVATION_ACTIVE_ARMS" ]]; then
        read -r -a INNOVATION_H16_ARMS <<< "control $INNOVATION_ACTIVE_ARMS"
        .venv/bin/python artifacts/experiments/20261012_oracle/innovation_eval.py --updates 0 3000 --horizon 16 --samples 4 --batch 4 --arms "${INNOVATION_H16_ARMS[@]}"
        CUDA_VISIBLE_DEVICES='' .venv/bin/python artifacts/experiments/20261012_oracle/innovation_analysis.py --updates 0 3000 --horizon 16
    else
        printf 'RECURSIVE READING DEFERRED: neither channel uses a varying current code; optimization diagnosis comes first\n'
    fi
    date '+SUCCESS %Y-%m-%d %H:%M:%S %Z'
} >> artifacts/eda/levers_logs/learned_innovation_eval.log 2>&1
