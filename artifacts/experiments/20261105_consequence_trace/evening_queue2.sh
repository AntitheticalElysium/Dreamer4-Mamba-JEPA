#!/bin/bash
# After evening_queue.sh: F2.1's interface at 3 and 16 steps (same histories), for the swing test (Codex 11-05: equal
# sampling budgets). Separate script so the running queue's file is never edited.
set -u
REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=$REPO/.venv/bin/python
OUT=$REPO/artifacts/experiments/20261105_consequence_trace/readings/evening
cd "$REPO"
while systemctl --user is-active -q trace-evening-queue; do sleep 30; done
while [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -c python)" -gt 0 ]; do sleep 20; done
echo "$(date '+%F %T') interface_f21_steps start" >> "$OUT/QUEUE.log"
systemd-run --user --unit=trace-evening-interface-f21 --wait --collect -p MemoryMax=12G -E MAMBA_DETERMINISTIC=1 \
  -E JAX_PLATFORMS=cpu --working-directory="$REPO" /bin/sh -c \
  "$PY -u -m artifacts.experiments.20261105_consequence_trace.interface_steps $OUT/interface_f21_steps.json artifacts/experiments/20261104_p1f21/run/ckpt_10.pt f21 > $OUT/interface_f21_steps.log 2>&1" \
  > "$OUT/interface_f21_steps.unit.txt" 2>&1
echo "$(date '+%F %T') interface_f21_steps exit $?" >> "$OUT/QUEUE.log"
