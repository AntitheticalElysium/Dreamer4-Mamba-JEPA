source artifacts/experiments/20260927_levers/lib.sh
# lane 4's tail, deferred behind the per-tile arms (lanes 9, 10): two 1.5 GB LDAD runs would block every 3.6 GB arm
until grep -q "LANE9_DONE" $LOGDIR/lanes.log && grep -q "LANE10_DONE" $LOGDIR/lanes.log; do sleep 60; done
D=artifacts/eda/levers_ldad_v1
full() { local out=$D/$1_lam$2; [ -f $out/step-010000.pt ] && return 0; rm -rf $out
         job ldad_$1_lam$2_full 2000 $PY $L/ldad_joint.py --variant $1 --lam $2; }
full raw 1
JAX_PLATFORMS=cpu job ldad_eval_full2 1800 $PY $L/ldad_eval.py
JAX_PLATFORMS=cpu job ldad_facts2 1800 $PY $L/ldad_facts.py
echo "$(date '+%F %T') LANE4_DONE" >> $LOGDIR/lanes.log
