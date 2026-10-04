source artifacts/experiments/20260927_levers/lib.sh
# E12 (predeclared 2026-10-01): FAIR's multistep rollout recipe (rolloutK), corrt Raw 18k, seeds 7/8 (+ rollout4 s7)
W=artifacts/eda/levers_tworlds_v1
until grep -qE "LANE26_DONE|FAILED missedscroll" $LOGDIR/lanes.log; do sleep 120; done
for arm in rollout2:7 rollout2:8 rollout4:7; do
  loss=${arm%:*}; seed=${arm#*:}; N=corrt_raw_${loss}_s${seed}_u18000
  [ -f $W/$N.pt ] || job tworld_$N 3100 $PY $L/tworld.py --head corrt --pool raw --loss $loss --seed $seed --updates 18000 --snapshots
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  [ -f $L/evals/${N}__w4_per_root.pt ] || job teval_${N}_w4 2000 $PY $L/teval.py $W/$N.pt --window 4
  [ -f $L/posprofile_$N.json ] || job posprofile_$N 1500 $PY $L/posprofile.py $W/$N.pt
  [ -f $L/blockwin_$N.json ] || job blockwin_$N 1500 $PY $L/blockwin.py $W/$N.pt
done
job compare_e12 0 $PY $L/compare.py corrt_raw_teacher_s7_u18000:corrt_raw_rollout2_s7_u18000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_rollout2_s8_u18000 corrt_raw_teacher_s7_u18000:corrt_raw_rollout4_s7_u18000 \
  corrt_raw_rollout2_s7_u18000:corrt_raw_rollout2_s8_u18000
R="$W/corrt_raw_rollout2_s7_u18000.pt $W/corrt_raw_rollout2_s8_u18000.pt $W/corrt_raw_rollout4_s7_u18000.pt"
[ -f $L/evals/driftanat_corrt_raw_rollout4_s7_u18000.json ] || job driftanat_e12 2000 $PY $L/driftanat.py $R
[ -f $L/evals/stochdiag_corrt_raw_rollout4_s7_u18000.json ] || job stochdiag_e12 2500 $PY $L/stochdiag.py $R
[ -f $L/evals/dpanel_corrt_raw_rollout4_s7_u18000.json ] || job dpanel_e12 3000 $PY $L/dpanel.py $R
until grep -qE "DONE deepeval_group3|FAILED deepeval_group3" $LOGDIR/lanes.log; do sleep 120; done
[ -f $L/evals/deep_corrt_raw_rollout4_s7_u18000.json ] || job deepeval_e12 3000 $PY $L/deepeval.py $R
job deepeval_compare_e12 0 $PY $L/deepeval.py --compare corrt_raw_teacher_s7_u18000:corrt_raw_rollout2_s7_u18000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_rollout2_s8_u18000 corrt_raw_teacher_s7_u18000:corrt_raw_rollout4_s7_u18000
echo "$(date '+%F %T') LANE27_DONE" >> $LOGDIR/lanes.log
