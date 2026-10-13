#!/usr/bin/env bash
# Durable, resumable observation-only training. Runtime output never edits NOTEBOOK.
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
    .venv/bin/python - <<'PY'
import json, hashlib
from pathlib import Path
p=Path('artifacts/eda/oracle_v1/learned_innovation/validation.json')
proof=json.loads(p.read_text())
assert all(proof['exact_streams'].values())
assert proof['initial_output_max_abs']==proof['native_belief_future_mutation_max_abs']==0
assert proof['hard_codes_exact'] and not proof['fixed_code_continuous_bypass']
for name,digest in proof['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest, name
PY
    .venv/bin/python artifacts/experiments/20261012_oracle/innovation_train.py --updates 3000 --batch 16
    date '+SUCCESS %Y-%m-%d %H:%M:%S %Z'
} >> artifacts/eda/levers_logs/learned_innovation_train.log 2>&1
