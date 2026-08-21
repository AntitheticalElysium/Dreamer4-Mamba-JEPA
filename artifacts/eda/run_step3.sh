#!/usr/bin/env bash
# Step 3: pixel control. Fork histories are re-captured with frames first, since the
# latent-only capture cannot feed a trainable encoder.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu

LAST_OK=1
stage() {
  local name="$1"; shift
  echo "=== $name  $(date -Is) ===" | tee -a triage.log
  local started=$SECONDS
  if "$@" >"logs/$name.log" 2>&1; then
    echo "    ok    $((SECONDS - started))s" | tee -a triage.log; LAST_OK=1
  else
    echo "    FAIL  $((SECONDS - started))s  (see logs/$name.log)" | tee -a triage.log
    tail -25 "logs/$name.log" | tee -a triage.log; LAST_OK=0
  fi
}

mkdir -p logs
stage 07_fork_frames      "$PY" reproduce_fork_histories.py
stage 08_pixel_preflight  "$PY" train_damage_pixels.py --preflight
if [ "$LAST_OK" = "1" ]; then
  stage 09_train_pixels   "$PY" train_damage_pixels.py --steps 20000
else
  echo "    skipping 09_train_pixels: preflight failed" | tee -a triage.log
fi
for milestone in 005000 010000 020000; do
  if [ -f "damage_pixels/model_${milestone}.pt" ]; then
    stage "10_eval_pixels_${milestone}" "$PY" evaluate_damage_pixels.py \
      --model "damage_pixels/model_${milestone}.pt"
  fi
done
echo "step 3 finished $(date -Is)" >> triage.log
