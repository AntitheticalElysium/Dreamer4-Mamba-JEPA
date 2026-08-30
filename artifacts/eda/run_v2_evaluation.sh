#!/usr/bin/env bash
# Waits for the training chain, then scores both arms on every gate that matters.
# Separate process on purpose: the pipeline script is running and bash reads scripts by
# byte offset, so it cannot be extended in place.
set -u
cd "$(dirname "$0")"
R=../..
PY=$R/.venv/bin/python
export PYTHONPATH=$R
say() { echo "[$(date +%H:%M:%S)] == $*"; }

say "waiting for the training chain"
while pgrep -f "[r]un_v2_pipeline" > /dev/null; do sleep 120; done
for arm in attention mamba; do
  [ -f "v2_direct_${arm}/world.pt" ] || { say "v2_direct_${arm} never finished; stopping"; exit 1; }
done
say "both arms trained; scoring"

for arm in attention mamba; do
  say "death transfer, $arm"
  $PY evaluate_death_transfer.py --folder "v2_direct_${arm}" --tag "v2_${arm}" > /dev/null 2>&1 \
      || say "death transfer FAILED for $arm"
  say "multi-step, $arm"
  $PY evaluate_multistep_forks.py --folder "v2_direct_${arm}" --tag "v2_${arm}" > /dev/null 2>&1 \
      || say "multistep FAILED for $arm"
  say "regression gates, $arm"
  $PY evaluate_production_1b.py --world "v2_direct_${arm}/world.pt" --tag "v2_${arm}" \
      > /dev/null 2>&1 || say "regression gates FAILED for $arm"
done

say "T versus M"
$PY - <<'EOF'
import json
from pathlib import Path
import numpy as np

def get(name):
    try: return json.load(open(name))
    except FileNotFoundError: return None

D = json.load(open("death_transfer_production_s2.json"))
lethal = np.array(D["per_root_lethal"])
print(f'{"":<26}{"overall":>9}{"escape":>8}{"trap":>7}{"d1 NSE":>9}{"d2 NSE":>9}'
      f'{"actMSE":>9}{"fork NSE":>10}{"cosine":>8}')
for tag, name in (("production_s2", "production (seed 2)"), ("broad_uniform_s2", "broad-uniform"),
                  ("v2_attention", "v2 Direct-T"), ("v2_mamba", "v2 Direct-M")):
    d = get(f"death_transfer_{tag}.json"); m = get(f"multistep_{tag}.json")
    r = get(f"production_1b_evaluation_{tag}.json")
    if d is None: continue
    v = np.array(d["per_root_pred"])
    row = f'  {name:<24}{v.mean():>9.3f}{v[lethal<=2].mean():>8.3f}{v[lethal>=14].mean():>7.3f}'
    row += (f'{m["per_depth"][0]["energy_weighted_nse"]:>9.3f}'
            f'{m["per_depth"][1]["energy_weighted_nse"]:>9.3f}') if m else f'{"":>18}'
    row += (f'{r["fork"]["action_mse"]:>9.5f}{r["fork"]["nse"]:>10.4f}'
            f'{r["fork"]["cosine"]:>8.4f}') if r else f'{"":>27}'
    print(row)
print("\n  d1/d2 NSE are alive-only, energy-weighted (lower better)")
EOF
say "evaluation complete"
