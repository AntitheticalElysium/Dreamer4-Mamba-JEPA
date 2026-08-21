#!/usr/bin/env bash
# Phase-1B translation diagnostic: does the 64-slot Z* make action consequences
# learnable by the ordinary Direct latent-dynamics objective? Idempotent and
# resumable at every stage.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi; }

while [ ! -f forkset_s1_n64/manifest.json ]; do sleep 60; done

# one lambda, measured once, applied to both geometries
[ -f lambda_preflight.json ] || stage 95_lambda_preflight "$PY" preflight_lambda.py
LAM=$("$PY" -c "
import json
d = json.load(open('lambda_preflight.json'))
r = [v['ratio'] for v in d.values()]
print(f'{sum(r)/len(r):.4f}')")
echo "lambda = $LAM (fixed, identical in both arms)" | tee -a triage.log

for n in 32 64; do
  [ -f "phase1b_s1_n${n}/training_report.json" ] || \
    stage "96_phase1b_n${n}" "$PY" train_phase1b_fork.py --n-latents $n --suffix s1 \
      --lam "$LAM" --steps 20000 --out "phase1b_s1_n${n}"
done
echo "phase1b finished $(date -Is)" >> triage.log
