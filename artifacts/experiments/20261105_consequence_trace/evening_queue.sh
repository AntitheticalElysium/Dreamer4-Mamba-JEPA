#!/bin/bash
# 11-05 evening (user: run the glimpse check before F3 and the 16-step F2 interface): after F2.1's evaluation is stopped
# at uptake (trace-sampler-queue ends) and the GPU is free: glimpse_check on F2 then F2.1, interface_steps on F2.
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings/evening
mkdir -p "$OUT"; cd "$REPO"
log() { echo "$(date '+%F %T') $*" >> "$OUT/QUEUE.log"; }
log "waiting for trace-sampler-queue (F2.1 evaluation to reach uptake)"
while systemctl --user is-active -q trace-sampler-queue; do sleep 30; done
while [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)" -gt 0 ]; do sleep 20; done
run() {                                                                  # run NAME MODULE ARGS...
  local name=$1; shift
  log "$name start"
  systemd-run --user --unit="trace-evening-$name" --wait --collect -p MemoryMax=12G -E MAMBA_DETERMINISTIC=1 \
    -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c "$PY -u -m $* > $OUT/$name.log 2>&1" > "$OUT/$name.unit.txt" 2>&1
  log "$name exit $?; $(grep -h 'Service runtime\|Memory peak' "$OUT/$name.unit.txt" | tr -s ' ' | tr '\n' ';')"
}
run glimpse_f2 artifacts.experiments.20261105_consequence_trace.glimpse_check $OUT/glimpse_f2.json artifacts/experiments/20261103_p1f2/run/ckpt_10.pt
run glimpse_f21 artifacts.experiments.20261105_consequence_trace.glimpse_check $OUT/glimpse_f21.json artifacts/experiments/20261104_p1f21/run/ckpt_10.pt
run interface_f2_steps artifacts.experiments.20261105_consequence_trace.interface_steps $OUT/interface_f2_steps.json
log "done"
