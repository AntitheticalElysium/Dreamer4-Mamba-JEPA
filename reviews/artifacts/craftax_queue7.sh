#!/usr/bin/env bash
# D045 ablation: widen the predictor's view of the post-dynamics state.
#   k1. tf=0.0, self-prediction only, spatial_agent  -> pairs with j1 (0.474)
#   k2. tf=0.5, all losses,            spatial_agent -> pairs with i1 (0.404)
# Single axis: jepa_predictor_context, 64-d pooled -> 384-d (spatial+agent).
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
stage k1_spatialpred_tf000_jepaonly \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --predictor-context spatial_agent \
    --terminal-fraction 0.0 --reward-weight 0.0 --continuation-weight 0.0 \
    --tag spatialpred_tf0.00_jepaonly --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_spatialpred_tf000_jepaonly.json
stage k2_spatialpred_tf050_all \
  $PY -u reviews/artifacts/craftax_timecourse.py \
    --predictor-context spatial_agent \
    --terminal-fraction 0.5 \
    --tag spatialpred_tf0.50_all --world-steps 2500 --ladder 0,1000,2500 \
    --output reviews/artifacts/craftax_spatialpred_tf050_all.json
echo "=== [$(date -u +%H:%M:%S)] QUEUE7 COMPLETE ===" | tee -a "$LOGDIR/queue.log"
