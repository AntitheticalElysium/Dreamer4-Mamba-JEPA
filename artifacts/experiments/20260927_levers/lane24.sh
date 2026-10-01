source artifacts/experiments/20260927_levers/lib.sh
# E11 (2026-10-01): randomness vs model in imagined error, by content class (stochdiag.py; readings declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
[ -f $L/evals/stochdiag_corrt_raw_selffed_s7.json ] || job stochdiag 2500 $PY $L/stochdiag.py \
  $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_suffix_s7_u18000.pt \
  $W/corrt_raw_suffix_s8_u18000.pt $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt $W/corrt_raw_teacher_s7_u18000_at6000.pt \
  $W/corrt_raw_noise_s7.pt $W/corrt_raw_selffed_s7.pt
echo "$(date '+%F %T') LANE24_DONE" >> $LOGDIR/lanes.log
