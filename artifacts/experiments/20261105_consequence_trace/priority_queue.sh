#!/bin/bash
# Explicit user order: smoke -> F2 -> F2.1 -> resume unfinished evaluations.
set -eu
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
cd "$REPO"
/bin/bash artifacts/experiments/20261105_consequence_trace/launch.sh --before-evaluation
"$PY" - <<'EOF'
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path('artifacts/experiments/20261105_consequence_trace/readings/QUEUE.json')
d = json.loads(p.read_text())
d.update(evaluation='resuming after successful traces', updated=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d, indent=1))
EOF
set +e
systemd-run --user --unit=saccade-p1f21-eval-recovery4 --wait --collect \
  --working-directory="$REPO" -p MemoryHigh=10G -p MemoryMax=11G -p OOMPolicy=stop \
  -p "ExecStopPost=$PY -m artifacts.experiments.20261104_p1f21.resume_evaluation --unit-exit" \
  -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E OMP_NUM_THREADS=4 -E MKL_NUM_THREADS=4 -E MALLOC_ARENA_MAX=2 \
  "$PY" -u -m artifacts.experiments.20261104_p1f21.resume_evaluation
rc=$?
set -e
"$PY" - "$rc" <<'EOF'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
p = Path('artifacts/experiments/20261105_consequence_trace/readings/QUEUE.json')
d = json.loads(p.read_text())
d.update(evaluation=f'exit {sys.argv[1]}', updated=datetime.now(timezone.utc).isoformat())
p.write_text(json.dumps(d, indent=1))
EOF
exit "$rc"
