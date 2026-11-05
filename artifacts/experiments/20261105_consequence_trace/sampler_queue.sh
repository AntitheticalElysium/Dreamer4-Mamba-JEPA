#!/bin/bash
# Stage B3 (PLAN.md) queue, user 11-05 ("proceed with all 3, executive authority"): (1) let F2.1's evaluation finish
# its running m05 stage, then pause it; (2) sampler_steps on F2, then F2.1; (3) resume the evaluation with Codex's
# recovery command and stop it when it reaches uptake (deferred: CPU 2 h 40 min, last in the queue, kept for later;
# completed stages persist). Steps (2)-(3) run whatever (2)'s exit codes.
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings/sampler_steps
QS=$REPO/artifacts/experiments/20261104_p1f21/readings/STATUS.json
R=$REPO/artifacts/experiments/20261105_consequence_trace/readings
mkdir -p "$OUT"; cd "$REPO"
log() { echo "$(date '+%F %T') $*" >> "$OUT/QUEUE.log"; }
qget() { "$PY" -c "import json; d=json.load(open('$QS')); print(d.get('current'), d['steps'].get('m05', {}).get('phase'))" 2>/dev/null; }
log "waiting for m05 to complete (now: $(qget))"
while systemctl --user is-active -q saccade-p1f21-eval-recovery5 && [ "$(qget | cut -d' ' -f2)" != complete ]; do sleep 20; done
log "m05: $(qget); pausing evaluation"
systemctl --user stop trace-stone-queue.service saccade-p1f21-eval-recovery5.service
while [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)" -gt 0 ]; do sleep 10; done
for w in f2 f21; do
  [ $w = f2 ] && ck=artifacts/experiments/20261103_p1f2/run/ckpt_10.pt || ck=artifacts/experiments/20261104_p1f21/run/ckpt_10.pt
  log "$w start"
  systemd-run --user --unit="trace-steps-$w" --wait --collect -p MemoryMax=12G -E MAMBA_DETERMINISTIC=1 \
    -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c \
    "$PY -u -m artifacts.experiments.20261105_consequence_trace.sampler_steps $OUT/$w $ck $R/$w $R/stone/$w > $OUT/$w.log 2>&1" \
    > "$OUT/$w.unit.txt" 2>&1
  log "$w exit $?; $(grep -h 'Service runtime\|Memory peak' "$OUT/$w.unit.txt" | tr -s ' ' | tr '\n' ';')"
done
log "evaluation resume (uptake deferred)"
systemd-run --user --unit=saccade-p1f21-eval-recovery6 --collect \
  --working-directory="$REPO" -p MemoryHigh=10G -p MemoryMax=11G -p OOMPolicy=stop \
  -p "ExecStopPost=$PY -m artifacts.experiments.20261104_p1f21.resume_evaluation --unit-exit" \
  -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
  -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E OMP_NUM_THREADS=4 -E MKL_NUM_THREADS=4 -E MALLOC_ARENA_MAX=2 \
  "$PY" -u -m artifacts.experiments.20261104_p1f21.resume_evaluation
sleep 30
while systemctl --user is-active -q saccade-p1f21-eval-recovery6; do
  if [ "$(qget | cut -d' ' -f1)" = uptake ]; then
    systemctl --user stop saccade-p1f21-eval-recovery6.service
    log "reached uptake: stopped (deferred by user 11-05; resume later with Codex's worker)"; exit 0
  fi
  sleep 20
done
log "evaluation unit ended before uptake: $(qget)"
