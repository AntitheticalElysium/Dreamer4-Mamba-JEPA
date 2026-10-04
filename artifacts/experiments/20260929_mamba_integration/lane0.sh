#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
OUT=artifacts/eda/spatial_pool_ldad1_v1
STAGE=artifacts/eda/spatial_pool_ldad1_v1.stage
CKPT=artifacts/eda/levers_ldad_v1/raw_lam1/step-010000.pt
if [ -e "$OUT/pool.json" ]; then
  echo 'lambda-1 pool already finalized; refusing overwrite'
  exit 0
fi
if [ -e "$STAGE" ]; then
  echo 'staging path exists; refusing overwrite without inspection'
  exit 2
fi
job pool_ldad1 2400 "$PY" "$L/tc_pool.py" "$CKPT" "$STAGE"
"$PY" - <<'PY'
import hashlib,json,os
from pathlib import Path
import torch
p=Path('artifacts/eda/spatial_pool_ldad1_v1.stage')
j=json.loads((p/'pool.json').read_text())
assert j['windows']==32647 and j['main']==24576 and j['terminal']==8071
assert j['encoder_sha256']==hashlib.sha256(Path('artifacts/eda/levers_ldad_v1/raw_lam1/step-010000.pt').read_bytes()).hexdigest()
base=torch.load('artifacts/eda/spatial_pool_v1/pool.pt',weights_only=False,mmap=True)
new=torch.load(p/'pool.pt',weights_only=False,mmap=True)
for k in ('ids','actions','reward_led','reward_valid','alive','dh','terminal'):
    if isinstance(base[k],torch.Tensor): assert torch.equal(base[k],new[k]),k
    else: assert base[k]==new[k],k
manifest={'state':'complete','source_pool_sha256':j['source_pool_sha256'],'encoder_sha256':j['encoder_sha256'],
          'pool_sha256':j['pool_sha256'],'id_and_label_arrays_equal':True,
          'encoder_script_sha256':hashlib.sha256(Path('artifacts/experiments/20260927_levers/tc_pool.py').read_bytes()).hexdigest()}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest),flush=True)
PY
mv "$STAGE" "$OUT"
echo "$(date '+%F %T') LANE_MAMBA_0_DONE" >> "$LOGDIR/lanes.log"
