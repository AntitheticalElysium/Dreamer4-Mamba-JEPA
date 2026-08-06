#!/usr/bin/env bash
# Preregistered Craftax encoder-LR transfer, full pipeline, SIGReg, narrow
# spatial interaction, and paired executed-control queue. Results are not
# committed or interpreted by this script.
set -u
set -o pipefail

REPO=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
cd "$REPO"
PY=.venv/bin/python
LOGDIR=outputs/d4_mamba_jepa/queue9
GRID=reviews/artifacts/lr_objective_grid
EXECUTED=reviews/artifacts/craftax_executed_lr
REPLAY_SHA=7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b
mkdir -p "$LOGDIR" "$GRID" "$EXECUTED"

exec 9>"$LOGDIR/queue9.lock"
if ! flock -n 9; then
  echo "queue9 is already running" >&2
  exit 1
fi
echo "$$" > "$LOGDIR/queue9.pid"

export JAX_PLATFORMS=cpu
export PYTHONHASHSEED=0

stage () {
  local name="$1"
  shift
  echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] START $name ===" \
    | tee -a "$LOGDIR/queue9.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] OK    $name ===" \
      | tee -a "$LOGDIR/queue9.log"
  else
    local code=$?
    echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] FAIL  $name code=$code ===" \
      | tee -a "$LOGDIR/queue9.log"
  fi
}

anchor () {
  local tag="$1"
  shift
  stage "$tag" "$PY" -u reviews/artifacts/craftax_encoder_anchor.py \
    --tag "$tag" --output-dir "$GRID" --world-steps 2500 \
    --ladder 0,1000,2500 --ema-steps 20000 --batch-size 8 "$@"
}

# First establish transfer to the exact failed all-heads/sampler recipe.
anchor transfer_ema_full_s20260727 \
  --seed 20260727 --encoder-lr 1e-4 --backend transformer \
  --anticollapse ema --predictor-context pooled_agent \
  --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
  --continuation-weight 1 --device cuda
anchor transfer_ema_slow_s20260727 \
  --seed 20260727 --encoder-lr 6e-6 --backend transformer \
  --anticollapse ema --predictor-context pooled_agent \
  --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
  --continuation-weight 1 --device cuda

# Start the longest primary job early. BC and imagination retain their original
# 1e-4 head optimizers and freeze the world; only world.encoder is at 6e-6.
stage slowenc_full_pipeline \
  "$PY" -u -m d4_mamba_jepa.craftax_run \
    --replay-sha256 "$REPLAY_SHA" \
    --output-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 \
    --world-steps 20000 --bc-steps 3000 --actor-steps 500 \
    --encoder-lr 6e-6 --seed 20260727 \
    --backends transformer,mamba2 --device cuda

stage slowenc_representation_oracle \
  "$PY" -u reviews/artifacts/craftax_oracle_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 \
    --arms t_jepa,m_jepa \
    --output reviews/artifacts/craftax_slowenc_oracle.json \
    --device cuda

# Complete the paired actual-recipe transfer replicates.
for seed in 20260728 20260729; do
  anchor "transfer_ema_full_s${seed}" \
    --seed "$seed" --encoder-lr 1e-4 --backend transformer \
    --anticollapse ema --predictor-context pooled_agent \
    --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
    --continuation-weight 1 --device cuda
  anchor "transfer_ema_slow_s${seed}" \
    --seed "$seed" --encoder-lr 6e-6 --backend transformer \
    --anticollapse ema --predictor-context pooled_agent \
    --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
    --continuation-weight 1 --device cuda
done

# Higher-dimensional SIGReg mechanism grid, cleanly matched to the completed
# EMA anchor grid. This is diagnostic, not a candidate promotion.
for seed in 20260727 20260728 20260729; do
  anchor "sigreg_clean_full_s${seed}" \
    --seed "$seed" --encoder-lr 1e-4 --backend transformer \
    --anticollapse sigreg --predictor-context pooled_agent \
    --terminal-fraction 0 --jepa-weight 1 --reward-weight 0 \
    --continuation-weight 0 --device cuda
  anchor "sigreg_clean_slow_s${seed}" \
    --seed "$seed" --encoder-lr 6e-6 --backend transformer \
    --anticollapse sigreg --predictor-context pooled_agent \
    --terminal-fraction 0 --jepa-weight 1 --reward-weight 0 \
    --continuation-weight 0 --device cuda
done

# Narrow spatial x slow-LR interaction only; no repeat of the rejected grid.
for seed in 20260727 20260728 20260729; do
  anchor "spatial_slow_s${seed}" \
    --seed "$seed" --encoder-lr 6e-6 --backend transformer \
    --anticollapse ema --predictor-context spatial_agent \
    --terminal-fraction 0 --jepa-weight 1 --reward-weight 0 \
    --continuation-weight 0 --device cuda
done

# One actual-recipe SIGReg interaction pair. It is intentionally a one-seed
# diagnostic and cannot promote an architecture.
anchor sigreg_actual_full_s20260727 \
  --seed 20260727 --encoder-lr 1e-4 --backend transformer \
  --anticollapse sigreg --predictor-context pooled_agent \
  --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
  --continuation-weight 1 --device cuda
anchor sigreg_actual_slow_s20260727 \
  --seed 20260727 --encoder-lr 6e-6 --backend transformer \
  --anticollapse sigreg --predictor-context pooled_agent \
  --terminal-fraction 0.5 --jepa-weight 1 --reward-weight 1 \
  --continuation-weight 1 --device cuda

# Same fixed fresh environment seeds for old full-LR and new slow-LR policies.
# Old checkpoints require an explicit implementation-drift waiver because the
# current launch commit adds provenance-only/default-preserving code.
stage executed_full_t \
  "$PY" -u reviews/artifacts/craftax_achievement_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_expert_v1 --arm t_jepa \
    --output "$EXECUTED/full_t.json" --seed-start 100000 \
    --episodes 30 --max-steps 2500 --context 8 --mode sample \
    --allow-implementation-drift --device cuda
stage executed_full_m \
  "$PY" -u reviews/artifacts/craftax_achievement_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_expert_v1 --arm m_jepa \
    --output "$EXECUTED/full_m.json" --seed-start 100000 \
    --episodes 30 --max-steps 2500 --context 8 --mode sample \
    --allow-implementation-drift --device cuda
stage executed_slow_t \
  "$PY" -u reviews/artifacts/craftax_achievement_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 --arm t_jepa \
    --output "$EXECUTED/slow_t.json" --seed-start 100000 \
    --episodes 30 --max-steps 2500 --context 8 --mode sample \
    --device cuda
stage executed_slow_m \
  "$PY" -u reviews/artifacts/craftax_achievement_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 --arm m_jepa \
    --output "$EXECUTED/slow_m.json" --seed-start 100000 \
    --episodes 30 --max-steps 2500 --context 8 --mode sample \
    --device cuda

stage aggregate_results \
  "$PY" -u reviews/artifacts/craftax_lr_summary.py

echo "=== [$(date -u +%Y-%m-%dT%H:%M:%SZ)] QUEUE9 COMPLETE ===" \
  | tee -a "$LOGDIR/queue9.log"
