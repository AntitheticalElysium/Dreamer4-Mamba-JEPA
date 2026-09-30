#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_SHORT6K_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
[ -f artifacts/experiments/20260929_mamba_integration/cache_audit.json ] || { echo "evaluation token cache audit missing"; exit 2; }
A=artifacts/eda/levers_mamba_integration_v1
for pool in raw ldad1 ldad10; do
  name=int_corrg_${pool}_suffix_s7_fmamba_u18000_at6000
  [ -f "$L/evals/${name}__readout_v2.json" ] || job mamba_eval_${pool}_short6k 2800 "$PY" "$L/teval.py" "$A/$name.pt"
done
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('artifacts/experiments/20260929_mamba_integration/eval6k_source.json')
files=['artifacts/experiments/20260927_levers/teval.py','artifacts/experiments/20260927_levers/tworld.py',
       'artifacts/experiments/20260929_mamba_integration/short_train.py']
p.write_text(json.dumps({x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in files},indent=2)+'\n')
PY
echo "$(date '+%F %T') LANE_MAMBA_EVAL6K_DONE" >> "$LOGDIR/lanes.log"
