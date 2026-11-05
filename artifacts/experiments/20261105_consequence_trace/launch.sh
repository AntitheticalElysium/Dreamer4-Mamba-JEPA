#!/bin/bash
# Stage B queue (PLAN.md, "Run"; user 11-05: launch automatically after F2.1's evaluation queue). Waits until F2.1's
# STATUS.json says complete and no python job holds the GPU (any other end state: stops without launching), then runs,
# each in its own capped unit under MAMBA_DETERMINISTIC=1: a 40-episode GPU smoke, F2, F2.1's selected checkpoint.
# The chain stops at the first failure. State, exit codes, runtimes and peaks: readings/QUEUE.json.
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings
STATUS=$REPO/artifacts/experiments/20261104_p1f21/STATUS.json
PY=$REPO/.venv/bin/python
F2=artifacts/experiments/20261103_p1f2/run/ckpt_10.pt
mkdir -p "$OUT"

state() {                                                    # state KEY VALUE -> readings/QUEUE.json
  "$PY" - "$OUT/QUEUE.json" "$1" "$2" <<'EOF'
import json, sys, time
path, key, val = sys.argv[1:4]
try:
    d = json.load(open(path))
except (OSError, ValueError):
    d = {}
d[key] = val
d["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(d, open(path, "w"), indent=1)
EOF
}
phase() { "$PY" -c "import json; print(json.load(open('$STATUS')).get('phase', '?'))" 2>/dev/null || echo "?"; }
gpu_python() { nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python; }
wait_gpu() { while [ "$(gpu_python)" -gt 0 ]; do sleep 60; done; }

step() {                                                     # step NAME CKPT [N_EPISODES]
  local name=$1 ckpt=$2 n=${3:-} peak=$OUT/$1.gpu_peak_mib rc sampler
  wait_gpu
  state "$name" "running since $(date '+%F %T')"
  echo 0 > "$peak"
  ( while true; do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
      [ "$m" -gt "$(cat "$peak")" ] && echo "$m" > "$peak"; sleep 5; done ) &
  sampler=$!
  systemd-run --user --unit="trace-$name" --wait --collect -p MemoryMax=16G -E MAMBA_DETERMINISTIC=1 \
    -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c \
    "$PY -u -m artifacts.experiments.20261105_consequence_trace.trace $OUT/$name $ckpt $n > $OUT/$name.log 2>&1" \
    > "$OUT/$name.unit.txt" 2>&1
  rc=$?
  kill "$sampler"
  state "$name" "exit $rc at $(date '+%F %T'); GPU peak (whole card) $(cat "$peak") MiB; $(grep -h 'Service runtime\|Memory peak' "$OUT/$name.unit.txt" | tr -s ' ' | tr '\n' ';')"
  return $rc
}

state queue "waiting for F2.1's evaluation (STATUS.json phase: $(phase))"
while true; do
  p=$(phase)
  [ "$p" = complete ] && [ "$(gpu_python)" -eq 0 ] && break
  [ "$p" = failed ] && { state queue "not launched: F2.1 STATUS.json phase 'failed'"; exit 1; }   # '?' = mid-write: wait
  sleep 60
done
F21=$("$PY" -c "import json; print('artifacts/experiments/20261104_p1f21/run/ckpt_%02d.pt' % json.load(open('$STATUS'))['selection']['checkpoint'])")
state f21_evaluation_complete_seen "$(date '+%F %T')"
state f21_checkpoint "$F21"
step smoke_gpu "$F2" 40 || { state queue "stopped: GPU smoke failed"; exit 1; }
step f2 "$F2" || { state queue "stopped: F2 failed"; exit 1; }
step f21 "$F21" || { state queue "stopped: F2.1 failed"; exit 1; }
state queue "done $(date '+%F %T')"
