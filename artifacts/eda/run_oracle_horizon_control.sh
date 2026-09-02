#!/usr/bin/env bash
# S80's oracle with the simulator standing in for the world model, at the horizon Direct
# trains (2) and at 16. The world is perfect by construction, so a win at 16 isolates the
# horizon as the constraint on the critic rather than the dynamics.
#
# Run on the v2 attention arm, not the pinned Stage-A checkpoints: those predate S85 and
# no longer load into `World` (2-layer MLP readout against the current direct_mixer), and
# the v2 encoder and BC are the ones whose actors this campaign diagnosed.
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for horizon in 2 16; do
  out=artifacts/oracle_horizon_h${horizon}
  if [ -f "$out/report.json" ]; then echo "[$(date +%H:%M:%S)] == oracle h=$horizon already done"; continue; fi
  echo "[$(date +%H:%M:%S)] == oracle h=$horizon"
  $PY artifacts/run_oracle_phase3.py --horizon "$horizon" --out "$out" --latents 64 \
    --phase1a artifacts/eda/capacity6k/n64d16_s1/encoder_006000.pt \
    --encoder-report artifacts/eda/capacity6k/n64d16_s1/training_report.json \
    --phase2 artifacts/eda/v2_phase2_attention/phase2_final.pt \
    || { echo "[$(date +%H:%M:%S)] == FAILED oracle h=$horizon"; exit 1; }
done
echo "[$(date +%H:%M:%S)] == oracle horizon control complete"
