#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_SHORT12K_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
[ -f artifacts/experiments/20260929_mamba_integration/cache_audit.json ] || { echo "evaluation token cache audit missing"; exit 2; }
"$PY" - <<'PY'
import hashlib,json
from pathlib import Path
expected=json.loads(Path('artifacts/experiments/20260929_mamba_integration/eval6k_source.json').read_text())
for path,sha in expected.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
PY
A=artifacts/eda/levers_mamba_integration_v1
for pool in raw ldad1 ldad10; do
  name=int_corrg_${pool}_suffix_s7_fmamba_u18000_at12000
  [ -f "$L/evals/${name}__readout_v2.json" ] || job mamba_eval_${pool}_short12k 2800 "$PY" "$L/teval.py" "$A/$name.pt"
done
echo "$(date '+%F %T') LANE_MAMBA_EVAL12K_DONE" >> "$LOGDIR/lanes.log"
