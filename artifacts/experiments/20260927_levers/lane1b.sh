source artifacts/experiments/20260927_levers/lib.sh
# lane 1's tail: seed-8 TC-vs-Raw now; the direct seed-8 arm deferred behind the per-tile fix and stage-2 lanes
W=artifacts/eda/levers_tworlds_v1
for name in residual_raw_suffix_s8 residual_tc_suffix_s8; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_tc_s8 0 $PY $L/compare.py residual_raw_suffix_s8:residual_tc_suffix_s8 residual_raw_suffix_s7:residual_raw_suffix_s8 residual_tc_suffix_s7:residual_tc_suffix_s8
until grep -q "LANE9_DONE" $LOGDIR/lanes.log && grep -q "LANE10_DONE" $LOGDIR/lanes.log; do sleep 60; done
[ -f $W/direct_raw_suffix_s8.pt ] || job tworld_direct_raw_suffix_s8 3000 $PY $L/tworld.py --head direct --pool raw --loss suffix --seed 8
[ -f $L/evals/direct_raw_suffix_s8_per_root.pt ] || job teval_direct_raw_suffix_s8 2000 $PY $L/teval.py $W/direct_raw_suffix_s8.pt
echo "$(date '+%F %T') LANE1_DONE" >> $LOGDIR/lanes.log
