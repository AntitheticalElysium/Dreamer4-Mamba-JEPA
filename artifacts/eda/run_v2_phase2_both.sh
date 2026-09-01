#!/usr/bin/env bash
# Phase 2 for both arms, then re-score depth-2 to see whether it survived.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }

for arm in attention mamba; do
  if [ -f "v2_phase2_${arm}/world.pt" ]; then say "phase 2 $arm already done"; continue; fi
  say "phase 2 $arm"
  $PY run_v2_phase2.py --arm "$arm" || { say "PHASE2 $arm FAILED"; exit 1; }
done

for arm in attention mamba; do
  say "depth-2 after phase 2, $arm"
  $PY evaluate_multistep_forks.py --folder "v2_phase2_${arm}" --tag "p2_${arm}" >/dev/null 2>&1 \
      || say "multistep FAILED for $arm"
  $PY evaluate_death_transfer.py --folder "v2_phase2_${arm}" --tag "p2_${arm}" >/dev/null 2>&1 \
      || say "death FAILED for $arm"
done

say "did phase 2 preserve the repair?"
$PY - <<'EOF'
import json
import numpy as np
print(f'{"":<22}{"d1":>8}{"d2":>8}   {"death":>7}{"escape":>8}{"trap":>7}')
D = json.load(open("death_transfer_production_s2.json")); lethal = np.array(D["per_root_lethal"])
for tag, name in (("production_s0","production"),("v2_attention","T after 1B"),
                  ("p2_attention","T after 2"),("v2_mamba","M after 1B"),("p2_mamba","M after 2")):
    try: m = json.load(open(f"multistep_{tag}.json"))["per_depth"]
    except FileNotFoundError: continue
    row = f'  {name:<20}{m[0]["energy_weighted_nse"]:>8.3f}{m[1]["energy_weighted_nse"]:>8.3f}'
    try:
        v = np.array(json.load(open(f"death_transfer_{tag}.json"))["per_root_pred"])
        row += f'   {v.mean():>7.3f}{v[lethal<=2].mean():>8.3f}{v[lethal>=14].mean():>7.3f}'
    except FileNotFoundError: pass
    print(row)
EOF
say "complete"
