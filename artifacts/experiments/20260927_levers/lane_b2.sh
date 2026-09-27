#!/usr/bin/env bash
# Lane B, revised after raw lambda 10 (prediction loss 6x the paired canonical run by update 1,500; LDAD accuracy at
# its identifiability ceiling). Lambda is screened at 2,000 updates, paired with the canonical step-2000 checkpoints;
# a full 10k TC run at the chosen lambda follows separately. Idempotent.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
D=artifacts/eda/levers_ldad_v1
screen() { # variant lam
  local out=$D/$1_lam$2_screen2000
  if [ -f $out/step-002000.pt ]; then return; fi
  rm -rf $out
  $PY $L/ldad_joint.py --variant $1 --lam $2 --steps 2000
}
screen raw 1
screen raw 0.1
screen tc 1
screen tc 10
JAX_PLATFORMS=cpu $PY $L/ldad_eval.py --screens
$PY $L/context_cont.py
[ -f $L/memory_use.json ] || $PY $L/memory_use.py
JAX_PLATFORMS=cpu $PY $L/ldad_eval.py
echo LANE_B2_DONE
