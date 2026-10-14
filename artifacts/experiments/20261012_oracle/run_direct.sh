#!/usr/bin/env bash
# One resumable two-arm study; confirmation is conditional on physical validity.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_AUTOTUNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/artifacts/eda/oracle_v1/direct_generator/runtime/triton_cache"
DIRECT_SCRIPT=artifacts/experiments/20261012_oracle
DIRECT_OUT=artifacts/eda/oracle_v1/direct_generator
mkdir -p artifacts/eda/levers_logs
exec >> artifacts/eda/levers_logs/direct_generator_20261014.log 2>&1
date '+START %Y-%m-%d %H:%M:%S %Z'
[[ -f "$DIRECT_OUT/validation.json" ]]
.venv/bin/python "$DIRECT_SCRIPT/direct_train.py" --updates 3000 --batch 16
for update in 0 500 1500 3000; do
    .venv/bin/python "$DIRECT_SCRIPT/direct_eval.py" --updates "$update" --samples 64 --batch 4
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$DIRECT_SCRIPT/direct_analysis.py" analyze \
        --directory "$DIRECT_OUT/eval_exploratory_u$(printf '%04d' "$update")"
done
DIRECT_CONFIRM=$(CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/eda/oracle_v1/direct_generator/eval_exploratory_u3000/analysis.json')
print('yes' if json.loads(p.read_text())['allows_confirmation'] else 'no')
PY
)
if [[ "$DIRECT_CONFIRM" == yes ]]; then
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$DIRECT_SCRIPT/direct_fresh.py" \
        --exploratory-result "$DIRECT_OUT/eval_exploratory_u3000/analysis.json"
    .venv/bin/python "$DIRECT_SCRIPT/direct_eval.py" --cohort fresh --updates 3000 --samples 64 --batch 4
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$DIRECT_SCRIPT/direct_analysis.py" analyze \
        --directory "$DIRECT_OUT/eval_fresh_u3000" --fresh "$DIRECT_OUT/fresh/fresh.pt"
else
    printf 'Fresh confirmation withheld: final exploratory physical/distribution criteria did not all pass.\n'
fi
date '+COMPLETE %Y-%m-%d %H:%M:%S %Z'
