#!/usr/bin/env bash
# Encoder-anchor ablation. tf=0, JEPA-only, EMA ramp pinned to 20,000 steps so a
# 2,500-update run is a genuine prefix of the schedule that actually failed.
# 3 encoder conditions x 3 seeds. Full oracle reports + checkpoints retained.
#   full   : enc_lr 1e-4  (current, unanchored)
#   slow   : enc_lr 6e-6  (Dreamer-CDP timescale separation, 66.7x)
#   frozen : enc_lr 0     (V-JEPA 2-AC / Dreamer 4 style; oracle score is
#                          constant BY CONSTRUCTION -- the informative number
#                          there is dev_cosine, i.e. whether self-prediction is
#                          achievable without moving the encoder)
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue
PY=.venv/bin/python
mkdir -p "$LOGDIR"
for seed in 20260727 20260728 20260729; do
  for cond in full:1e-4 slow:6e-6 frozen:0.0; do
    name="${cond%%:*}"; lr="${cond##*:}"
    tag="anchor_${name}_s${seed}"
    echo "=== [$(date -u +%H:%M:%S)] START $tag ===" | tee -a "$LOGDIR/queue.log"
    if $PY -u reviews/artifacts/craftax_encoder_anchor.py \
         --tag "$tag" --encoder-lr "$lr" --seed "$seed" \
         > "$LOGDIR/$tag.log" 2>&1; then
      echo "=== [$(date -u +%H:%M:%S)] OK    $tag ===" | tee -a "$LOGDIR/queue.log"
    else
      echo "=== [$(date -u +%H:%M:%S)] FAIL  $tag ===" | tee -a "$LOGDIR/queue.log"
    fi
  done
done
echo "=== [$(date -u +%H:%M:%S)] QUEUE8 COMPLETE ===" | tee -a "$LOGDIR/queue.log"
