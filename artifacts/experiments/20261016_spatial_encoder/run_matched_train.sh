#!/usr/bin/env bash
# MATCHED_ENCODER_PREREG.md training: MoP-s1 pool (only if R0a passed) -> wait for the frozen-state stage-B training (GPU strictly
# sequential) -> R0b (Raw recipe, 1,000 updates, objective within 5% of the historical log) -> the MoP arm (36k run, snapshots;
# stopped after 12k by run_matched_eval.sh if the pre-declared continuation rule says so).
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python; H=artifacts/experiments/20261016_spatial_encoder; M=artifacts/eda/matched_encoder_v1
export TRITON_F32_DEFAULT=ieee PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -c "import json,sys; sys.exit(0 if json.load(open('$M/r0a.json'))['R0a_pass'] else 1)" || { echo R0A_FAILED; exit 1; }
[ -f artifacts/eda/spatial_pool_mop_s1_v1/pool.pt ] || $PY $H/matched_pool.py build
while systemctl --user is-active -q frozen-B; do sleep 120; done
[ -f artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_di_u1000.pt ] || \
  $PY $H/matched_d_launcher.py --head corrt --pool raw --loss teacher --seed 7 --updates 1000 --backbone fmamba --state-every 2000 --recon deltairis
$PY - <<'PYEOF'
import json, torch
new = torch.load('artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_di_u1000.pt', map_location='cpu', weights_only=False)['history']
old = torch.load('artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_di_u36000_at6000.pt', map_location='cpu', weights_only=False)['history']
o = {r['update']: r['objective'] for r in old}; n = {r['update']: r['objective'] for r in new}
rel = {u: abs(n[u] - o[u]) / o[u] for u in (500, 1000)}
res = {'historical': {u: o[u] for u in (500, 1000)}, 'rerun': {u: n[u] for u in (500, 1000)}, 'relative_diff': rel,
       'tolerance': 0.05, 'R0b_pass': all(v <= 0.05 for v in rel.values())}
json.dump(res, open('artifacts/eda/matched_encoder_v1/r0b.json', 'w'), indent=1); print(json.dumps(res))
raise SystemExit(0 if res['R0b_pass'] else 1)
PYEOF
$PY $H/matched_d_launcher.py --head corrt --pool mop_s1 --loss teacher --seed 7 --updates 36000 --backbone fmamba --snapshots --state-every 2000 --recon deltairis
echo matched_train_done
