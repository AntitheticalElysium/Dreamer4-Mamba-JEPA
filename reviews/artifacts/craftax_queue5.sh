#!/usr/bin/env bash
# Paired sampler control (2,500 updates, probes at 0/1000/2500).
#   I1. terminal_fraction=0.5 -- the current sampler, RERUN fresh because
#       nominally identical endpoints have shown material per-target variance.
#   I2. terminal_fraction=0.0 -- forced terminal windows removed. Every loss,
#       weight, schedule and architecture setting is otherwise identical, and
#       both arms share the same torch init seed.
# If removing forced terminal windows prevents the health spike and the task
# state erosion, the sampler is established as the primary cause.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue
PY=.venv/bin/python

stage () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$LOGDIR/queue.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] OK    $name ===" | tee -a "$LOGDIR/queue.log"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $name ===" | tee -a "$LOGDIR/queue.log"
  fi
}

stage i1_termfrac_050 \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --terminal-fraction 0.5 --tag termfrac_0.50 \
    --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_sampler_termfrac050.json

stage i2_termfrac_000 \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --terminal-fraction 0.0 --tag termfrac_0.00 \
    --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_sampler_termfrac000.json

echo "=== [$(date -u +%H:%M:%S)] QUEUE5 COMPLETE ===" | tee -a "$LOGDIR/queue.log"
