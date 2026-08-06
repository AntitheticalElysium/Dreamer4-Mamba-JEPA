#!/usr/bin/env bash
# Corrected paired T/M comparison at the slow encoder LR.
#
# Both defects fixed since queue9:
#   * the mamba2 substitution now runs LAST in D4LiteWorld.__init__, so the
#     JEPA predictor and all three projection MLPs are bit-identical across
#     backends (16 shared tensors previously differed at init)
#   * train_craftax_bc / train_craftax_imagination reseed before building their
#     heads, so BC and value heads no longer inherit backend-dependent RNG
#
# Everything else matches the queue9 slow arms exactly: enc_lr 6e-6, all other
# world params 1e-4, 20k world / 3k BC / 500 actor, seed 20260727. Executed
# evaluation reuses the SAME 30 environment seeds (100000..100029), context 8,
# sampled mode, policy_seed_base 7000000.
#
# Purpose: answer whether M-JEPA's imagination failure survives a valid
# single-axis comparison, now that the arms actually share an initialization.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue10
OUT=outputs/d4_mamba_jepa/craftax_fixedinit_slow
PY=.venv/bin/python
SHA=7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b
mkdir -p "$LOGDIR"

stage () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$LOGDIR/queue10.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] OK    $name ===" | tee -a "$LOGDIR/queue10.log"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $name ===" | tee -a "$LOGDIR/queue10.log"
  fi
}

# A. Cheapest first (audit priority #1): does the existing slow-T gain survive
#    a different policy-sampling seed base? Re-evaluates queue9 checkpoints only.
for base in 7100000 7200000; do
  for arm in t_jepa m_jepa; do
    stage "resample_${arm}_${base}" \
      $PY -u reviews/artifacts/craftax_achievement_run.py \
        --run-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 --arm "$arm" \
        --policy-seed-base "$base" \
        --output reviews/artifacts/craftax_executed_lr/resample_${arm}_${base}.json
  done
done

# B. The corrected paired training run.
stage b_fixedinit_slow_pipeline \
  $PY -u -m d4_mamba_jepa.craftax_run \
    --replay-sha256 "$SHA" --output-dir "$OUT" \
    --encoder-lr 6e-6 --seed 20260727 --backends transformer,mamba2

# C. Executed evaluation on the SAME 30 seeds as queue9.
for arm in t_jepa m_jepa; do
  stage "c_executed_${arm}" \
    $PY -u reviews/artifacts/craftax_achievement_run.py \
      --run-dir "$OUT" --arm "$arm" \
      --output reviews/artifacts/craftax_executed_lr/fixedinit_slow_${arm}.json
done

echo "=== [$(date -u +%H:%M:%S)] QUEUE10 COMPLETE ===" | tee -a "$LOGDIR/queue10.log"
