source artifacts/experiments/20260927_levers/lib.sh
# E9 (2026-10-01): decision panel (H1/H2 action-dependent death) for per-tile worlds, opened blocks 55k-56k
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
# one invocation per group: references are fitted once per invocation; finished worlds are skipped on relaunch
[ -f $L/evals/dpanel_corrt_raw_teacher_s7_u18000_at6000.json ] || job dpanel_group1 3000 $PY $L/dpanel.py $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_teacher_s7_u18000.pt \
  $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt $W/corrt_raw_suffix_s7.pt $W/corrt_raw_teacher_s7_u18000_at6000.pt
until grep -q "LANE18B_DONE" $LOGDIR/lanes.log; do sleep 60; done
[ -f $L/evals/dpanel_corrt_raw_teacher_s8.json ] || job dpanel_group2 3000 $PY $L/dpanel.py $W/corrt_raw_noise_s7.pt $W/corrt_raw_selffed_s7.pt $W/corrt_raw_suffix_s8.pt \
  $W/corrt_raw_teacher_s8.pt
job dpanel_compare 0 $PY $L/dpanel.py --compare corrt_raw_teacher_s7_u18000:corrt_raw_suffix_s7_u18000 \
  corrt_raw_suffix_s7_u18000:int_corrg_raw_suffix_s7_fmamba_u18000 corrt_raw_teacher_s7_u18000_at6000:corrt_raw_suffix_s7 \
  corrt_raw_teacher_s7_u18000_at6000:corrt_raw_noise_s7 corrt_raw_teacher_s7_u18000_at6000:corrt_raw_selffed_s7 \
  corrt_raw_teacher_s8:corrt_raw_suffix_s8
echo "$(date '+%F %T') LANE19B_DONE" >> $LOGDIR/lanes.log
