source artifacts/experiments/20260927_levers/lib.sh
# E15 (2026-10-02, the user's go-ahead): the sealed H16 evaluation (E10 stage 3: deepeval.py, judge block 62,000-62,399, rules of
# 8c969101) of the budget worlds, added to the sealed set here before any of them is read, with the seed-8 18k comparator (lane23
# group3's teacher world). Readings, declared before running (36k worlds; comparator = same-seed teacher u18000):
#   h16_carried     gen16 - prior16 resolved > 0 at both seeds (deep_<36k>.json contrasts)
#   h16_usable      transfer16 - prior16 resolved > 0 at both seeds
#   budget_gen16    (36k - 18k) gen16 paired difference resolved > 0 at both seeds AND larger than the head-seed spread
#                   (max - min of the 3 per-seed values) of both worlds (dpanel / deepeval intervals omit head-fit variance;
#                   check_headseeds)
#   budget_transfer16  the same for transfer16
#   H1 / H4 reported.
W=artifacts/eda/levers_tworlds_v1
[ -f $L/evals/deep_corrt_raw_teacher_s7_u36000.json ] || job deepeval_36k_s7 3000 $PY $L/deepeval.py $W/corrt_raw_teacher_s7_u36000.pt
[ -f $L/evals/deep_corrt_raw_teacher_s8_u18000.json ] || job deepeval_18k_s8 3000 $PY $L/deepeval.py $W/corrt_raw_teacher_s8_u18000.pt
[ -f $L/evals/deep_corrt_raw_teacher_s8_u36000.json ] || job deepeval_36k_s8 3000 $PY $L/deepeval.py $W/corrt_raw_teacher_s8_u36000.pt
job deepeval_compare_budget 0 $PY $L/deepeval.py --compare corrt_raw_teacher_s7_u36000:corrt_raw_teacher_s7_u18000 \
  corrt_raw_teacher_s8_u36000:corrt_raw_teacher_s8_u18000
echo "$(date '+%F %T') LANE42_DONE" >> $LOGDIR/lanes.log
