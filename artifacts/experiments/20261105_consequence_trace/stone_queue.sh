#!/bin/bash
# Stage B2 (PLAN.md): stone trace on F2 then F2.1, then resume F2.1's paused evaluation with Codex's recovery command
# (priority_queue.sh), whatever the traces' exit codes. User 11-05: "pause F2.1 to run this, it's more important".
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings/stone
F2=artifacts/experiments/20261103_p1f2/run/ckpt_10.pt
F21=artifacts/experiments/20261104_p1f21/run/ckpt_10.pt
mkdir -p "$OUT"
cd "$REPO"
while [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)" -gt 0 ]; do sleep 20; done
for w in f2 f21; do
  [ $w = f2 ] && ck=$F2 || ck=$F21
  echo "$w start $(date '+%F %T')" >> "$OUT/QUEUE.log"
  systemd-run --user --unit="trace-stone-$w" --wait --collect -p MemoryMax=12G -E MAMBA_DETERMINISTIC=1 \
    -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c \
    "$PY -u -m artifacts.experiments.20261105_consequence_trace.trace $OUT/$w $ck 0 stone > $OUT/$w.log 2>&1" \
    > "$OUT/$w.unit.txt" 2>&1
  echo "$w exit $? $(date '+%F %T'); $(grep -h 'Service runtime\|Memory peak' "$OUT/$w.unit.txt" | tr -s ' ' | tr '\n' ';')" >> "$OUT/QUEUE.log"
done
echo "evaluation resume $(date '+%F %T')" >> "$OUT/QUEUE.log"
systemd-run --user --unit=saccade-p1f21-eval-recovery5 --wait --collect \
  --working-directory="$REPO" -p MemoryHigh=10G -p MemoryMax=11G -p OOMPolicy=stop \
  -p "ExecStopPost=$PY -m artifacts.experiments.20261104_p1f21.resume_evaluation --unit-exit" \
  -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E OMP_NUM_THREADS=4 -E MKL_NUM_THREADS=4 -E MALLOC_ARENA_MAX=2 \
  "$PY" -u -m artifacts.experiments.20261104_p1f21.resume_evaluation
echo "evaluation exit $? $(date '+%F %T')" >> "$OUT/QUEUE.log"
