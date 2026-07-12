#!/bin/bash
# Mamba-param sweep on the NEW trained-encoder JEPA baseline (Q1 re-run in the correct latent).
# Each config trains encoder+Mamba JOINTLY (the real setup) on 40-file PPO frames, 4000 steps.
# Baseline (2/512/64) is jepacnn_v1 (WM ratio 0.742, act-acc 0.436). We vary ONE knob at a time.
# The question: does Mamba capacity now lower WM ratio in the higher-info latent (unlike frozen-space Q1)?
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/jepacnn

run () {  # tag  n_layers d_model d_state
  tag=$1; nl=$2; dm=$3; ds=$4
  echo "===== $tag : layers=$nl d_model=$dm d_state=$ds ====="
  python -u jepa/train_jepa_wm.py --nfiles 40 --steps 4000 --eval_every 1000 \
    --n_layers $nl --d_model $dm --d_state $ds --out_tag "$tag" > "$L/$tag.log" 2>&1 \
    && grep -aE "^params|step   4000" "$L/$tag.log" | tail -2 \
    || echo "$tag FAILED (see $L/$tag.log)"
}

run dstate128 2 512 128
run dstate256 2 512 256
run layers4   4 512 64
run dmodel768 2 768 64
echo "===== MAMBA SWEEP DONE ====="
