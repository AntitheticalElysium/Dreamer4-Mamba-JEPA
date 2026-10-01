source artifacts/experiments/20260927_levers/lib.sh
# E11b (2026-10-01): position vs content in the deterministic drift (driftanat.py; readings declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
until grep -qE "LANE24_DONE|FAILED stochdiag" $LOGDIR/lanes.log; do sleep 60; done
[ -f $L/evals/driftanat_corrt_raw_selffed_s7.json ] || job driftanat 2000 $PY $L/driftanat.py \
  $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_suffix_s7_u18000.pt \
  $W/corrt_raw_suffix_s8_u18000.pt $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt $W/corrt_raw_noise_s7.pt $W/corrt_raw_selffed_s7.pt
echo "$(date '+%F %T') LANE25_DONE" >> $LOGDIR/lanes.log
