#!/usr/bin/env bash
# The full campaign, in order. Each stage is resumable and refuses to clobber existing output.
#
#   1 joint     fresh Raw and TC from update zero on the merged corpus, with the declared fork term
#   2 bridge    readout, BC, reward/continuation heads, then H2 -> H16 on the same world
#   3 actor     PMPO imagination actor, world and prior frozen
#   4 evaluate  512 shared DEV seeds at the native cap, actor versus its own BC
#
# Read PREDECLARATION.md before looking at any number produced here.
set -euo pipefail
cd "$(dirname "$0")/../../.."
ROOT=$PWD
D=artifacts/experiments/20260920_m4_bridge_campaign
# FLATTENED is the selected condition: fork roots enter as ordinary corpus episodes through the
# normal JointSampler path, at the same batch size and with no extra per-update work. Set
# MODE=grouped to run Direct's branch contract instead (fork_mass 0.2, ~136 extra differentiable
# transitions per update). The two answer different questions; see PLAN.md.
MODE=${MODE:-flat}
OUT=${OUT:-artifacts/lewm_m4_paired_$MODE}
if [ "$MODE" = "flat" ]; then
  RECIPES=(--raw-recipe "$D/recipes/lewm_mamba_raw_m4_flat.json"
           --tc-recipe  "$D/recipes/lewm_mamba_tc_m4_flat.json")
  CORPUS=(--corpus artifacts/craftax_expert_store_v1 artifacts/craftax_support_v2
                    artifacts/craftax_forks_flat_v1)
else
  RECIPES=(--raw-recipe "$D/recipes/lewm_mamba_raw_m4.json"
           --tc-recipe  "$D/recipes/lewm_mamba_tc_m4.json")
  CORPUS=(--corpus artifacts/craftax_expert_store_v1 artifacts/craftax_support_v2)
fi
export TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu PYTHONPATH=.:$D
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

echo "== stage 1: paired joint training from update zero =="
$PY $D/train_joint_pair.py --out "$OUT" --stop-at "${JOINT_STEPS:-10000}" \
    "${RECIPES[@]}" "${CORPUS[@]}"

for arm in raw tc; do
  JOINT="$OUT/$arm/joint/step-$(printf '%06d' "${JOINT_STEPS:-10000}").pt"
  echo "== stage 2: bridge ($arm) =="
  # Deliberately NOT --corpus "${CORPUS[@]}": the flattened forks carry fabricated zero rewards on
  # their history transitions and are not logged behaviour, so they must never reach the reward,
  # continuation or BC heads. They train the world only.
  $PY $D/bridge.py --checkpoint "$JOINT" --out "$OUT/$arm/bridge"
  BRIDGE=$(ls -1 "$OUT/$arm/bridge"/bridge_*.pt | sort | tail -1)

  echo "== stage 3: actor ($arm) =="
  $PY $D/actor.py --bridge "$BRIDGE" --encoder-checkpoint "$JOINT" --out "$OUT/$arm/actor"
  ACTOR=$(ls -1 "$OUT/$arm/actor"/actor_*.pt | sort | tail -1)

  echo "== stage 4: real Craftax, actor versus its own BC ($arm) =="
  $PY $D/evaluate.py --actor "$ACTOR" --joint "$JOINT" --out "$OUT/$arm/evaluation"
done

echo "== campaign complete =="
for arm in raw tc; do
  $PY -c "
import json;d=json.load(open('$OUT/$arm/evaluation/evaluation.json'))
p=d['primary'];print(f\"$arm: actor-BC achievements {p['achievements_gap']:+.3f} \"
      f\"{p['achievements_interval']} beats={p['actor_beats_own_bc']}\")"
done
