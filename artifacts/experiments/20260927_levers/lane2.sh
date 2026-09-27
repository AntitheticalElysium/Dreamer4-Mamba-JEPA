#!/usr/bin/env bash
# Lane 2: Delta-JEPA lambda screens (paired with canonical step 2000), LDAD evaluations, matched-compute context arms.
source artifacts/experiments/20260927_levers/lib.sh
D=artifacts/eda/levers_ldad_v1
screen() { # variant lam
  local out=$D/$1_lam$2_screen2000
  [ -f $out/step-002000.pt ] && return 0
  rm -rf $out
  job ldad_$1_lam$2_screen 2000 $PY $L/ldad_joint.py --variant $1 --lam $2 --steps 2000
}
screen raw 1
screen raw 0.1
screen tc 1
screen tc 10
JAX_PLATFORMS=cpu job ldad_eval_screens 1800 $PY $L/ldad_eval.py --screens
JAX_PLATFORMS=cpu job ldad_eval_full 1800 $PY $L/ldad_eval.py
job context_matched 2700 $PY $L/context_cont.py
job memory_use 1800 $PY $L/memory_use.py
echo "$(date '+%F %T') LANE2_DONE" >> $LOGDIR/lanes.log
