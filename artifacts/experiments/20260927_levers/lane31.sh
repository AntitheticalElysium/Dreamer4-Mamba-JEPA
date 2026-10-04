source artifacts/experiments/20260927_levers/lib.sh
# E14m (2026-10-02): is deterministic imagination "monotone"? (monotone.py; the categorical world also sampled)
W=artifacts/eda/levers_tworlds_v1
for w in categorical_raw_teacher_s7_K4096 corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000 direct_raw_suffix_s7; do
  [ -f $L/evals/monotone_$w.json ] || job monotone_$w 1500 $PY $L/monotone.py $W/$w.pt
done
echo "$(date '+%F %T') LANE31_DONE" >> $LOGDIR/lanes.log
