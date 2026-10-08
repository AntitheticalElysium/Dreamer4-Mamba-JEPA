source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 4: per-world drawable / substitution / ingredient table (health_evidence.py) on the lane97/98 files.
job hevid_main 1200 $PY $L/health_evidence.py main \
  artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000__w15__ext.pt \
  artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_L16b40_from36000__w15__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s8_L16b40_from36000__w15__ext.pt \
  artifacts/eda/health_chain_v1/e20_B_s7_fmamba_fromM16__w15__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at6000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at12000__w5__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at18000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at24000__w5__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at30000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000__w5__ext.pt \
  artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_u36000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_u50000__w5__ext.pt artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_u100000__w5__ext.pt
echo "$(date '+%F %T') LANE100_DONE" >> $LOGDIR/lanes.log
