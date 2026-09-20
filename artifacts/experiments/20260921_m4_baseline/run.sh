#!/usr/bin/env bash
# Canonical M4 baseline, Raw and TC. Idempotent: re-running resumes every stage in place.
#
#   1  paired-run   preflight, paired init, 2,000 updates per arm, G1, then 10,000
#   2  bridge H2    2,000 recursive updates per arm at H=2
#   STOP            H16 needs an identity-bound G2/G3 report. No evaluator produces one yet
#                   (STATUS.md:42), and a hand-written pass would be manufactured evidence.
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

say "BASELINE REACHED THE G2/G3 GATE."
say "Both arms have a 10,000-update joint world and a 2,000-update H2 bridge."
say "H16, the actor and real evaluation need an identity-bound bridge gate report that no"
say "evaluator currently produces. Build the G2/G3 evaluator from spec/lewm/EVALUATION.md."
