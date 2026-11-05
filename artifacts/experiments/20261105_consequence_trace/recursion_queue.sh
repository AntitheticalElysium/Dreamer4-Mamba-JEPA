#!/bin/bash
# NOT ARMED until the user's go (11-05 P2 discussion): F2's own-history pilot (Codex's recursion.py, parameterised copy)
# at 3 then 16 sampler steps, same 128 windows and seeds; after evening_queue2 (the 16-step interfaces), before F3.
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings/recursion
mkdir -p "$OUT"; cd "$REPO"
while systemctl --user is-active -q trace-evening-queue2; do sleep 30; done
while [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)" -gt 0 ]; do sleep 20; done
for n in 3 16; do
  echo "$(date '+%F %T') f2 steps $n start" >> "$OUT/QUEUE.log"
  systemd-run --user --unit="trace-recursion-$n" --wait --collect -p MemoryMax=12G -E MAMBA_DETERMINISTIC=1 \
    -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c \
    "$PY -u -m artifacts.experiments.20261105_consequence_trace.recursion_steps --ckpt artifacts/experiments/20261103_p1f2/run/ckpt_10.pt --out $OUT/f2_steps$n --steps $n > $OUT/f2_steps$n.log 2>&1" \
    > "$OUT/f2_steps$n.unit.txt" 2>&1
  echo "$(date '+%F %T') f2 steps $n exit $?; $(grep -h 'Service runtime\|Memory peak' "$OUT/f2_steps$n.unit.txt" | tr -s ' ' | tr '\n' ';')" >> "$OUT/QUEUE.log"
done
