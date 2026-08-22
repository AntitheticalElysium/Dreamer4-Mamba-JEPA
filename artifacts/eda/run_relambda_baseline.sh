#!/usr/bin/env bash
# One clean 64x16 MAE-Direct-Transformer baseline under the corrected protocol.
#
# Changed, and only this: the world backbone now receives the real causal action
# history (a_{t-1} convention) instead of the BOS/null token at every block, which
# also stops the ordinary teacher-forced term from conditioning its readout on a
# token the all-17 fork term never uses. Latents are untouched -- the encoder never
# sees an action -- so `forkset_s1fixb_n64` is a symlink to the existing forkset.
#
# Everything else is the established setup: same roots, same whole-seed split, same
# Direct-Attention world under the same initialization seed, same optimizer and draw
# streams, same 20k budget. Lambda is 1.4758, not 2.3761: the corrected calibration
# found the inherited weight over-weights the fork term 1.610x at initialisation
# (registered band [0.67, 1.50], so a fail) and 3.187x at the 20k model. 1.4758 is
# initialisation parity under exactly the convention that produced 2.3761 -- only the
# protocol it is measured under changed. Restarted from zero rather than resumed, per
# the registered rule: never change lambda mid-run.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

[ -f phase1b_s1fixb_n64/training_report.json ] || \
  stage E1_train_relam "$PY" train_phase1b_fork.py --n-latents 64 --suffix s1fixb \
    --lam 1.4758 --steps 20000 --out phase1b_s1fixb_n64
for m in 5000 10000 20000; do
  [ -f "phase1b_delta_s1fixb_n64_$(printf %06d $m).json" ] || \
    stage "E2_eval_${m}" "$PY" reevaluate_phase1b_delta.py --n-latents 64 --suffix s1fixb --milestone $m
done
stage E3_traj "$PY" reevaluate_phase1b_delta.py --trajectory --suffix s1fixb --n-latents 64 \
  --milestones 5000 10000 20000
stage E4_path "$PY" probe_direct_path.py --n-latents 64 --suffix s1fixb
echo "relambda baseline finished $(date -Is)" >> triage.log
