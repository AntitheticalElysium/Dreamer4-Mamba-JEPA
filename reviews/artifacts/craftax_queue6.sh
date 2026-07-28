#!/usr/bin/env bash
# Completes the sampler x objective 2x2 at 2,500 updates.
#   j1. terminal_fraction=0.0, SELF-PREDICTION ONLY (reward=continuation=0)
#       -- the contingency pre-declared for "erosion persists after the sampler fix".
#   j2. terminal_fraction=0.5, self-prediction only -- the matching cell, so the
#       sampler and objective axes are separable rather than confounded.
# Existing cells: i1 (0.5, all losses), i2 (0.0, all losses), h2 (0.5, heads only).
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
stage j1_tf000_jepaonly \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --terminal-fraction 0.0 --reward-weight 0.0 --continuation-weight 0.0 \
    --tag tf0.00_jepaonly --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_tf000_jepaonly.json
stage j2_tf050_jepaonly \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --terminal-fraction 0.5 --reward-weight 0.0 --continuation-weight 0.0 \
    --tag tf0.50_jepaonly --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_tf050_jepaonly.json
echo "=== [$(date -u +%H:%M:%S)] QUEUE6 COMPLETE ===" | tee -a "$LOGDIR/queue.log"
