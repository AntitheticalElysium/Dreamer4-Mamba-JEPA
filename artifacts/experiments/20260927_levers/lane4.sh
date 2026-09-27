#!/usr/bin/env bash
# Lane 4 (E4c): full 10k LDAD runs where the 2k screens cannot judge imagination (the world lags the encoder):
# TC lambda 10 (does LDAD restore TC's static terrain through imagination?) and Raw lambda 1 (a gentler balance).
source artifacts/experiments/20260927_levers/lib.sh
D=artifacts/eda/levers_ldad_v1
full() { local out=$D/$1_lam$2; [ -f $out/step-010000.pt ] && return 0; rm -rf $out
         job ldad_$1_lam$2_full 2000 $PY $L/ldad_joint.py --variant $1 --lam $2; }
full tc 10
full raw 1
JAX_PLATFORMS=cpu job ldad_eval_full2 1800 $PY $L/ldad_eval.py
JAX_PLATFORMS=cpu job ldad_facts2 1800 $PY $L/ldad_facts.py
echo "$(date '+%F %T') LANE4_DONE" >> $LOGDIR/lanes.log
