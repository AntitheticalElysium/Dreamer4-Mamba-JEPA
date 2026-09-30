source artifacts/experiments/20260927_levers/lib.sh
# E5h: evaluation-only -- ITC's binarized decoding on the soft copy heads (leak test: static rollouts drift 1.4-1.5x copy)
W=artifacts/eda/levers_tworlds_v1
for name in corrt_raw_suffix_s7 corrg_raw_suffix_s7_u18000 corr_raw_suffix_s7; do
  [ -f $L/evals/${name}__hard_per_root.pt ] || job teval_hard_$name 2000 $PY $L/teval.py $W/$name.pt --hard
done
job compare_hard 0 $PY $L/compare.py corrt_raw_suffix_s7:corrt_raw_suffix_s7__hard corrg_raw_suffix_s7_u18000:corrg_raw_suffix_s7_u18000__hard corr_raw_suffix_s7:corr_raw_suffix_s7__hard
echo "$(date '+%F %T') LANE12_DONE" >> $LOGDIR/lanes.log
