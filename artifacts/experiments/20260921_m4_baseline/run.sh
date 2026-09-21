#!/usr/bin/env bash
# Canonical M4 baseline, Raw and TC. Idempotent: re-running resumes every stage in place.
#
#   1  paired-run   preflight, paired init, 2,000 updates per arm, G1, then 10,000
#   2  bridge H2    2,000 recursive updates per arm at H=2
#   3  gate h2      the sealed G2/G3 report; H16 proceeds only if every component passes
#   4  bridge H16   8,000 more updates per arm
#   5  gate h16     the sealed report that authorizes the actor
#   6  actor screen 500 updates, then gate actor, then the 5,000 budget
#   7  evaluate     512 paired seeds, actor versus its own immutable BC
#
# DO NOT EDIT ANY FILE IN THE sources.py RUNTIME CLOSURE WHILE THIS RUNS. Every checkpoint hashes
# that closure, and a mid-flight edit makes the run unresumable and fails G1. Learned the hard way
# on 2026-09-21: four checkpoints recorded three different manifests and the run was discarded.
set -uo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu PYTHONPATH=.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
OUT=${OUT:-artifacts/lewm_m4_canonical}
DATA=(artifacts/craftax_expert_store_v1 artifacts/craftax_support_v2)
LOG=artifacts/experiments/20260921_m4_baseline

say() { echo "[$(date -Is)] $*" | tee -a "$LOG/driver.log"; }

say "stage 1: paired joint run -> G1 -> 10,000"
if [ -f "$OUT/pair.json" ]; then
  say "  resuming the existing sealed pair"
  $PY -m d4mj paired-run --dataset "${DATA[@]}" --out "$OUT" --resume
else
  $PY -m d4mj paired-run --dataset "${DATA[@]}" --out "$OUT"
fi
rc=$?
say "stage 1 rc=$rc"
[ $rc -ne 0 ] && { say "stage 1 did not complete; stopping"; exit $rc; }

for arm in raw tc; do
  say "stage 2: bridge H2, arm $arm"
  latest=$(ls -1 "$OUT/$arm/bridge"/step-*.pt 2>/dev/null | sort | tail -1)
  if [ -n "$latest" ]; then
    say "  resuming from $(basename "$latest")"
    $PY -m d4mj bridge --run "$OUT/$arm" --stop-after h2 --resume "$latest"
  else
    $PY -m d4mj bridge --run "$OUT/$arm" --stop-after h2
  fi
  rc=$?
  say "  bridge H2 $arm rc=$rc"
  [ $rc -ne 0 ] && { say "bridge H2 $arm failed; stopping"; exit $rc; }
done

for arm in raw tc; do
  say "stage 3: G2/G3 gate at H2, arm $arm"
  $PY -m d4mj gate --run "$OUT/$arm" --stage h2 --dataset "${DATA[@]}"
  rc=$?
  say "  gate h2 $arm rc=$rc"
  if [ $rc -ne 0 ]; then
    say "GATE FAILED for $arm. This is a result, not a bug: H16 is refused because a component"
    say "did not beat its declared baseline. Read $OUT/$arm/gates/h2/bridge_gate_h2.json."
    exit $rc
  fi
done

for arm in raw tc; do
  say "stage 4: bridge H16, arm $arm"
  latest=$(ls -1 "$OUT/$arm/bridge"/step-*.pt 2>/dev/null | sort | tail -1)
  $PY -m d4mj bridge --run "$OUT/$arm" --stop-after h16 --resume "$latest"       --gate "$OUT/$arm/gates/h2/bridge_gate_h2.json"
  rc=$?; say "  bridge H16 $arm rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  say "stage 5: gate at H16, arm $arm"
  $PY -m d4mj gate --run "$OUT/$arm" --stage h16 --dataset "${DATA[@]}"
  rc=$?; say "  gate h16 $arm rc=$rc"
  [ $rc -ne 0 ] && { say "H16 gate refused the actor for $arm"; exit $rc; }
done

for arm in raw tc; do
  say "stage 6: actor screen, arm $arm"
  $PY -m d4mj actor --run "$OUT/$arm" --stop-after screen       --bridge-gate "$OUT/$arm/gates/h16/bridge_gate_h16.json"
  rc=$?; say "  actor screen $arm rc=$rc"
  [ $rc -ne 0 ] && exit $rc
  $PY -m d4mj gate --run "$OUT/$arm" --stage actor
  rc=$?; say "  gate actor $arm rc=$rc"
  [ $rc -ne 0 ] && { say "actor screen gate refused the full budget for $arm"; exit $rc; }
  latest=$(ls -1 "$OUT/$arm/actor"/step-*.pt 2>/dev/null | sort | tail -1)
  $PY -m d4mj actor --run "$OUT/$arm" --stop-after budget --resume "$latest"       --bridge-gate "$OUT/$arm/gates/h16/bridge_gate_h16.json"       --actor-gate "$OUT/$arm/gates/actor/actor_gate.json"
  rc=$?; say "  actor budget $arm rc=$rc"
  [ $rc -ne 0 ] && exit $rc
done

for arm in raw tc; do
  say "stage 7: real Craftax, actor versus its own BC, arm $arm"
  $PY -m d4mj evaluate --run "$OUT/$arm"
  say "  evaluate $arm rc=$?"
done
say "BASELINE COMPLETE, END TO END."
