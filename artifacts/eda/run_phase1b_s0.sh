#!/usr/bin/env bash
# Phase-1B replication at the other Phase-1A tokenizer seed.
#
# Only the tokenizer seed changes: seed-offset 0 (seed 20260731) instead of the
# seed-offset 1 (20260732) pair already run. Same S82 roots, same whole-seed split,
# same Direct-Attention world under the same initialization seed, same optimizer and
# draw streams, same 20k budget, same fixed lambda, same probes and projection seeds.
#
# Stage 0 re-scores the already-finished s1 arms so both seeds carry the paired
# 64-minus-32 bootstrap; it also exercises the evaluator before the long jobs.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
LAM=2.3761   # identical to the s1 run, not re-measured

stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi; }

# --- stage 0: paired endpoint on the seed already finished --------------------
for n in 32 64; do
  stage "97_rescore_s1_n${n}" "$PY" reevaluate_phase1b_delta.py --n-latents $n --suffix s1
done
stage 97_paired_s1 "$PY" reevaluate_phase1b_delta.py --combine --suffix s1

# --- stage 1: encode the all-17 forkset under the s0 tokenizer ----------------
for n in 32 64; do
  [ -f "forkset_s0_n${n}/manifest.json" ] || \
    stage "98_encode_s0_n${n}" "$PY" encode_fork_dataset.py --n-latents $n --suffix s0 \
      --milestone 6000 --out "forkset_s0_n${n}"
done

# --- stage 2: the two dynamics arms ------------------------------------------
for n in 32 64; do
  [ -f "phase1b_s0_n${n}/training_report.json" ] || \
    stage "99_phase1b_s0_n${n}" "$PY" train_phase1b_fork.py --n-latents $n --suffix s0 \
      --lam "$LAM" --steps 20000 --out "phase1b_s0_n${n}"
done

# --- stage 3: the registered endpoints ---------------------------------------
for n in 32 64; do
  stage "99_eval_s0_n${n}" "$PY" reevaluate_phase1b_delta.py --n-latents $n --suffix s0
done
stage 99_paired_s0 "$PY" reevaluate_phase1b_delta.py --combine --suffix s0

echo "phase1b s0 replication finished $(date -Is)" >> triage.log
