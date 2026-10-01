source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-01: the 18k teacher-vs-suffix contrast rests on one init seed (E8 showed 0.057 seed spread at depth 16).
# Seed-8 18k pair, recipes identical to corrt_raw_{suffix,teacher}_s7_u18000; waits for the E9 panel's first group.
W=artifacts/eda/levers_tworlds_v1
until grep -qE "DONE dpanel_group1|FAILED dpanel_group1" $LOGDIR/lanes.log; do sleep 60; done
for loss in teacher suffix; do
  N=corrt_raw_${loss}_s8_u18000
  [ -f $W/$N.pt ] || job tworld_$N 3600 $PY $L/tworld.py --head corrt --pool raw --loss $loss --seed 8 --updates 18000 --snapshots
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  [ -f $L/posprofile_$N.json ] || job posprofile_$N 1500 $PY $L/posprofile.py $W/$N.pt
  [ -f $L/blockwin_$N.json ] || job blockwin_$N 1500 $PY $L/blockwin.py $W/$N.pt
done
job compare_seed8_18k 0 $PY $L/compare.py corrt_raw_suffix_s8_u18000:corrt_raw_teacher_s8_u18000 \
  corrt_raw_suffix_s7_u18000:corrt_raw_suffix_s8_u18000 corrt_raw_teacher_s7_u18000:corrt_raw_teacher_s8_u18000
echo "$(date '+%F %T') LANE22_DONE" >> $LOGDIR/lanes.log
