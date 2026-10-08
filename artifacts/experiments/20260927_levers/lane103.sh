source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 6: evidence table for the lane99 / lane102 worlds (L16 u500, E20 A, matched heads at 6k, M6 s8,
# generator-loss arms and baselines, E19 B / C).
until grep -q "LANE102_DONE" $LOGDIR/lanes.log; do sleep 20; done
job hevid_more 1200 $PY $L/health_evidence.py more \
  artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000_u500__w15__ext.pt artifacts/eda/health_chain_v1/e20_A_s7_fmamba_fromM16__w15__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_teacher_s8_fmamba_u36000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_u18000_at6000__w5__ext.pt \
  artifacts/eda/health_chain_v1/residual_raw_teacher_s7__w5__ext.pt artifacts/eda/health_chain_v1/direct_raw_suffix_s7__w5__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_suffix_s7__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_suffix_s7_gl__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_suffix_s7_u18000__w5__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_suffix_s7_gl_itc_u18000__w5__ext.pt artifacts/eda/health_chain_v1/e19_B_s7_fmamba_from36000__w5__ext.pt artifacts/eda/health_chain_v1/e19_C_s7_fmamba_from36000__w5__ext.pt
echo "$(date '+%F %T') LANE103_DONE" >> $LOGDIR/lanes.log
