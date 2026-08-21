#!/usr/bin/env bash
# Pre-H2 triage, queued. Each stage logs to its own file and records an exit code,
# so a failure stops that stage rather than the queue.
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
    echo "    ok    $((SECONDS - started))s" | tee -a triage.log
    LAST_OK=1
  else
    echo "    FAIL  $((SECONDS - started))s  (see logs/$name.log)" | tee -a triage.log
    tail -25 "logs/$name.log" | tee -a triage.log
    LAST_OK=0
  fi
}

mkdir -p logs
echo "triage started $(date -Is)" >> triage.log

# Stage 1 waits for the all-action latent collection already in flight.
while [ ! -f latent_forks/manifest.json ]; do sleep 30; done
echo "latent collection complete $(date -Is)" >> triage.log

stage 01_probe_delta_z          "$PY" probe_consequence.py
if [ -f damage_labels.npz ]; then
  echo "=== 02_damage_labels: already built, skipped ===" | tee -a triage.log
else
  stage 02_damage_labels        "$PY" build_damage_labels.py
fi
stage 03_fork_histories         "$PY" reproduce_fork_histories.py
stage 04_train_preflight        "$PY" train_damage_classifier.py --preflight
if [ "$LAST_OK" = "1" ]; then
  stage 05_train_damage         "$PY" train_damage_classifier.py --steps 20000
else
  echo "    skipping 05_train_damage: preflight failed" | tee -a triage.log
fi
for milestone in 005000 010000 020000; do
  if [ -f "damage_classifier/model_${milestone}.pt" ]; then
    stage "06_eval_${milestone}" "$PY" evaluate_damage_classifier.py \
      --model "damage_classifier/model_${milestone}.pt"
  fi
done

echo "triage finished $(date -Is)" >> triage.log
