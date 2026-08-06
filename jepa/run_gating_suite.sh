#!/bin/bash
# Gated-vs-ungated u_s-imagination experiment suite (matched settings).
# Two starts (from-random / from-BC) x two conditions (ungated / gated@0.9).
# Each: 80-iter online loop (per-iter EVAL + gate-fraction logged) + final stochastic eval vs random.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
source .venv/bin/activate
L=runs/gating                      # persistent repo-local logs (scratchpad UUIDs change per session)
mkdir -p "$L"
ITERS=80

run () {  # tag  extra-args...
  tag=$1; shift
  echo "===== RUN $tag ====="
  python -u jepa/train_online.py --out_tag "$tag" --iters $ITERS --collect 500 \
      --wm_updates 40 --agent_updates 20 "$@" > "$L/gate_$tag.log" 2>&1 || echo "$tag TRAIN FAILED"
  python -u jepa/eval_crafter.py --wm "ckpt/jepa_wm_$tag.pt" --agent "ckpt/jepa_agent_$tag.pt" \
      --episodes 16 > "$L/eval_$tag.log" 2>&1 || echo "$tag EVAL FAILED"
  echo "--- $tag final eval ---"; grep -E "POLICY|RANDOM" "$L/eval_$tag.log" || true
}

run rand_ungated
run rand_gated   --gate_pct 0.9
run bc_ungated   --agent_init ckpt/jepa_agent_bc.pt
run bc_gated     --agent_init ckpt/jepa_agent_bc.pt --gate_pct 0.9
echo "===== SUITE DONE ====="
