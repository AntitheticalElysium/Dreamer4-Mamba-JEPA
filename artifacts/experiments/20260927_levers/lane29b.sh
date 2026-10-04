source artifacts/experiments/20260927_levers/lib.sh
# E13c (2026-10-01): which component fails to learn consequences? consfit across saved heads / losses / budgets / backbones
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
job consfit_heads 2500 $PY $L/consfit.py $W/direct_raw_suffix_s7.pt $W/residual_raw_suffix_s7.pt $W/residual_raw_teacher_s7.pt \
  $W/gated_raw_suffix_s7.pt $W/corr_raw_suffix_s7.pt $W/corrg_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s7.pt \
  $W/corrt_raw_suffix_s7_gl.pt $W/corrt_raw_suffix_s7_gl_itc.pt $W/corrt_raw_suffix_s7_gl_itc_u18000.pt \
  $W/categorical_raw_teacher_s7_K4096.pt $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt $W/corrt_raw_rollout2_s7_u18000.pt
echo "$(date '+%F %T') LANE29B_DONE" >> $LOGDIR/lanes.log
