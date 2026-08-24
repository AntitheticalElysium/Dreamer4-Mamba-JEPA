#!/usr/bin/env bash
# One pooled, one-block Direct repair. The only substantive change is that the
# candidate action becomes a token mixed with the pooled spatial tokens through a
# single pre-norm self-attention block, instead of being broadcast identically over
# already-pooled features. Pool, action embedding, terminal projection Linear(256,32),
# tanh, encoder, loss, lambda and action history are all unchanged.
#
# abt0 is the control: MixerWorld calls super().__init__ first, so all 112 shared
# parameters initialise bit-identically, and the batch-draw and commit streams are
# seeded independently of the model. Verified, 0 differing tensors.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

[ -f phase1b_abm0_n64/training_report.json ] || \
  stage J1_train_mixer "$PY" train_phase1b_fork.py --n-latents 64 --suffix abm0 \
    --lam 1.4758 --steps 20000 --world-seed 0 --mixer --out phase1b_abm0_n64
for m in 5000 10000 20000; do
  [ -f "phase1b_delta_abm0_n64_$(printf %06d $m).json" ] || \
    stage "J2_eval_${m}" "$PY" reevaluate_phase1b_delta.py --n-latents 64 --suffix abm0 \
      --milestone $m --mixer
done
stage J3_traj "$PY" reevaluate_phase1b_delta.py --trajectory --suffix abm0 --n-latents 64 \
  --milestones 5000 10000 20000
stage J4_path "$PY" probe_direct_path.py --n-latents 64 --suffix abm0 --mixer
stage J5_matching "$PY" probe_action_matching.py --suffix abm0 --mixer
echo "mixer arm finished $(date -Is)" >> triage.log
