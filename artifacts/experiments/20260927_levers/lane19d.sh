source artifacts/experiments/20260927_levers/lib.sh
# E9 (2026-10-01): decision panel on the seed-8 18k pair, for the two-seed 18k recipe contrast; after lane23's first group.
W=artifacts/eda/levers_tworlds_v1
# 2026-10-02: run now (cheap, GPU idle while E14 is prepared); no wait on lane 23
[ -f $L/evals/dpanel_corrt_raw_suffix_s8_u18000.json ] || job dpanel_group4 3000 $PY $L/dpanel.py $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
job dpanel_compare2 0 $PY $L/dpanel.py --compare corrt_raw_teacher_s7_u18000:corrt_raw_suffix_s7_u18000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_suffix_s8_u18000 corrt_raw_teacher_s7_u18000_at6000:corrt_raw_suffix_s7 \
  corrt_raw_teacher_s8:corrt_raw_suffix_s8 corrt_raw_suffix_s7:direct_raw_suffix_s7 corrt_raw_suffix_s7:residual_raw_suffix_s7
echo "$(date '+%F %T') LANE19D_DONE" >> $LOGDIR/lanes.log
