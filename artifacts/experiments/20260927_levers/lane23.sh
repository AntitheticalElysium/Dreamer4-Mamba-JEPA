source artifacts/experiments/20260927_levers/lib.sh
# E10 stage 3 (2026-10-01): the H16 evaluation, rules committed before the judgement block 62,000-62,399 is read.
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
until grep -q "LANE20A_DONE" $LOGDIR/lanes.log && grep -q "LANE20B_DONE" $LOGDIR/lanes.log; do sleep 120; done
# 2026-10-01 18:53: E12 training (~3 GB) cannot share the GPU with deepeval (~2.1 GB): wait for it
until grep -qE "DONE tworld_corrt_raw_rollout4_s7_u18000|FAILED tworld_corrt_raw_rollout" $LOGDIR/lanes.log; do sleep 120; done
[ -f $L/evals/deep_int_corrg_raw_suffix_s7_fmamba_u18000.json ] || job deepeval_group1 3000 $PY $L/deepeval.py \
  $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_suffix_s7_u18000.pt $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt
# 2026-10-01 after the restart: the seed-8 18k pair (needed by E12's two-seed rule) before the 6k pairs
until grep -qE "LANE22_DONE" $LOGDIR/lanes.log; do sleep 120; done
[ -f $L/evals/deep_corrt_raw_suffix_s8_u18000.json ] || job deepeval_group3 3000 $PY $L/deepeval.py \
  $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
[ -f $L/evals/deep_corrt_raw_suffix_s8.json ] || job deepeval_group2 3000 $PY $L/deepeval.py \
  $W/corrt_raw_teacher_s7_u18000_at6000.pt $W/corrt_raw_suffix_s7.pt $W/corrt_raw_teacher_s8.pt $W/corrt_raw_suffix_s8.pt
job deepeval_compare 0 $PY $L/deepeval.py --compare corrt_raw_teacher_s7_u18000:corrt_raw_suffix_s7_u18000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_suffix_s8_u18000 corrt_raw_suffix_s7_u18000:int_corrg_raw_suffix_s7_fmamba_u18000 \
  corrt_raw_teacher_s7_u18000_at6000:corrt_raw_suffix_s7 corrt_raw_teacher_s8:corrt_raw_suffix_s8
echo "$(date '+%F %T') LANE23_DONE" >> $LOGDIR/lanes.log
