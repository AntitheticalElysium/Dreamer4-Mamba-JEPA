#!/usr/bin/env bash
# Sequential unattended run queue for the Craftax encoder investigation.
#
# Priority order is deliberate: close the verified geometric divergence from the
# pinned source FIRST (n_latents), then test the objective (T-BASE) at matched
# geometry. Every stage writes its own log and its own oracle JSON, so a failure
# in one stage does not lose the stages before it.
#
# Pre-declared, not selected after the fact:
#   A. n_latents 64 and 256 with d_bottleneck pinned at the paper's 16
#   B. oracle on both
#   C. T-BASE at n_latents=16 -- objective control at the EXACT baseline geometry
#   D. oracle on T-BASE@16
#   E. T-BASE at n_latents=64 -- objective x geometry interaction
#   F. oracle on T-BASE@64
set -u  # not -e: a failed stage must not abort the remaining ones

cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
LOGDIR=outputs/d4_mamba_jepa/queue
mkdir -p "$LOGDIR"

stage () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$LOGDIR/queue.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] OK    $name ===" | tee -a "$LOGDIR/queue.log"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $name (see $LOGDIR/$name.log) ===" \
      | tee -a "$LOGDIR/queue.log"
  fi
}

stage a_n_latents_ladder \
  $PY -u reviews/artifacts/craftax_capacity.py --grid 64:16,256:16

stage b_oracle_n_latents \
  $PY -u reviews/artifacts/craftax_oracle_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_capacity \
    --arms n_latents_64_d_bottleneck_16,n_latents_256_d_bottleneck_16 \
    --output reviews/artifacts/craftax_oracle_n_latents.json

stage c_tbase_n16 \
  $PY -u reviews/artifacts/craftax_tbase.py --n-latents 16

stage d_oracle_tbase_n16 \
  $PY -u reviews/artifacts/craftax_oracle_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_tbase \
    --arms t_base \
    --output reviews/artifacts/craftax_oracle_tbase16.json

stage e_tbase_n64 \
  $PY -u reviews/artifacts/craftax_tbase.py --n-latents 64

stage f_oracle_tbase_n64 \
  $PY -u reviews/artifacts/craftax_oracle_run.py \
    --run-dir outputs/d4_mamba_jepa/craftax_tbase \
    --arms t_base_n_latents_64 \
    --output reviews/artifacts/craftax_oracle_tbase64.json

echo "=== [$(date -u +%H:%M:%S)] QUEUE COMPLETE ===" | tee -a "$LOGDIR/queue.log"
