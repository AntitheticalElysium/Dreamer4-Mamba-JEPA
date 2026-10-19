#!/usr/bin/env bash
# One matched resumable event-factorization study. Root launches after proofs.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export TRITON_CACHE_AUTOTUNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$PWD/artifacts/eda/oracle_v1/paired_event/runtime/triton_cache"
PAIRED_SCRIPT=artifacts/experiments/20261012_oracle
PAIRED_OUT=artifacts/eda/oracle_v1/paired_event
mkdir -p artifacts/eda/levers_logs
exec >> artifacts/eda/levers_logs/paired_event_20261014.log 2>&1
date '+START %Y-%m-%d %H:%M:%S %Z'
[[ -f "$PAIRED_OUT/validation.json" ]]
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('artifacts/eda/oracle_v1/paired_event/validation.json')
q=json.loads(p.read_text())
assert q['passed'] is True and q['historical_direct_initial_state_exact']
assert q['shared_backbone_and_trunk_exact']
for name,expected in q['sources'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
assert all(q['resume_streams_exact'].values())
assert max(q['resume_endpoint_maxabs'].values())<1e-4
assert q['real_batch16_peak_allocated_gib']<6
for arm in ('direct','paired'):
    e=q['arms'][arm]
    assert e['noise_replay_maxabs']==0 and e['future_mutation_current_output_maxabs']==0
    assert e['independent_draw_output_maxabs']>0 and e['backbone_gradient_norm']>0
    assert all(x>0 for x in e['draw_gradient_norms']) and e['first_power_linear_scaling']
# The pre-main reference reading is binding evidence, never assumed sensible.
assert Path('artifacts/eda/oracle_v1/paired_event/eval_exploratory_u0000/analysis.json').is_file()
PY
.venv/bin/python "$PAIRED_SCRIPT/paired_train.py" --updates 3000 --batch 16
for update in 0 500 1500 3000; do
    .venv/bin/python "$PAIRED_SCRIPT/paired_eval.py" --updates "$update" --samples 64 --batch 4
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_analysis.py" analyze \
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
    CUDA_VISIBLE_DEVICES='' .venv/bin/python "$PAIRED_SCRIPT/paired_analysis.py" analyze \
        --directory "$PAIRED_OUT/eval_fresh_u3000" --fresh "$PAIRED_OUT/fresh/fresh.pt"
else
    printf 'Fresh confirmation withheld: final exploratory physical/distribution criteria did not all pass.\n'
fi
date '+COMPLETE %Y-%m-%d %H:%M:%S %Z'
