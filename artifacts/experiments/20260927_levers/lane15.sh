source artifacts/experiments/20260927_levers/lib.sh
# E5i: held-out learning curve of corrt to 36k updates (snapshots every 6k): is 18k still under-training?
until grep -q "LANE9_DONE" $LOGDIR/lanes.log; do sleep 60; done
W=artifacts/eda/levers_tworlds_v1
N=corrt_raw_suffix_s7_u36000
[ -f $W/$N.pt ] || job tworld_$N 3600 $PY $L/tworld.py --head corrt --pool raw --loss suffix --updates 36000 --snapshots
for u in 6000 12000 18000 24000 30000; do
  [ -f $L/evals/${N}_at${u}_per_root.pt ] || job teval_${N}_at$u 2000 $PY $L/teval.py $W/${N}_at$u.pt
done
[ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
echo "$(date '+%F %T') LANE15_DONE" >> $LOGDIR/lanes.log
