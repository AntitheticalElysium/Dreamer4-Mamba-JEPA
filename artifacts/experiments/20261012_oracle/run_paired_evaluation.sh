#!/usr/bin/env bash
# Resume completed-model evaluation; the frozen training program is untouched.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_AUTOTUNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/artifacts/eda/oracle_v1/paired_event/runtime/triton_cache"
PAIRED_SCRIPT=artifacts/experiments/20261012_oracle
PAIRED_OUT=artifacts/eda/oracle_v1/paired_event
mkdir -p artifacts/eda/levers_logs
exec >> artifacts/eda/levers_logs/paired_event_evaluation_resume_20261014.log 2>&1
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/eda/oracle_v1/paired_event/training/training.json')
x=json.loads(p.read_text())
assert x['completed'] and x['update']==3000
assert set(x['totals'])=={'direct','paired'}
assert all(t['steps']==3000 and t['targets']==720000 for t in x['totals'].values())
PY
for update in 0 500 1500 3000; do
    .venv/bin/python "$PAIRED_SCRIPT/paired_eval.py" --updates "$update" --samples 64 --batch 4
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_resume_analysis.py" \
        --directory "$PAIRED_OUT/eval_exploratory_u$(printf '%04d' "$update")"
done
.venv/bin/python "$PAIRED_SCRIPT/paired_diagnostics.py" export --updates 3000
CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_diagnostics.py" analyze --updates 3000
PAIRED_CONFIRM=$(CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('artifacts/eda/oracle_v1/paired_event/eval_exploratory_u3000/analysis.json')
print('yes' if json.loads(p.read_text())['allows_confirmation'] else 'no')
PY
)
if [[ "$PAIRED_CONFIRM" == yes ]]; then
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_diagnostics.py" collect-fresh \
        --exploratory-result "$PAIRED_OUT/eval_exploratory_u3000/analysis.json"
    .venv/bin/python "$PAIRED_SCRIPT/paired_eval.py" --cohort fresh --updates 3000 --samples 64 --batch 4
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_resume_analysis.py" \
        --directory "$PAIRED_OUT/eval_fresh_u3000" --fresh "$PAIRED_OUT/fresh/fresh.pt"
else
    printf 'Fresh confirmation withheld: registered coherent-completion conditions failed.\n'
fi
date '+EVALUATION COMPLETE %Y-%m-%d %H:%M:%S %Z'
