source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 22:50: oracle positive control at the unbiased row (window 10, output row 9) + CIs against M16 / C0.
until grep -q "LANE118_DONE" $LOGDIR/lanes.log; do sleep 30; done
for N in corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000_at500 corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000; do job hext_w10_$N 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$N.pt --window 10 --ext; done
job hevid_oracle_row9 1100 $PY $L/health_evidence.py oracle_row9 artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000_at500__w10__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000__w10__ext.pt
job hci_oracle 800 $PY $L/health_ci.py oracle M16=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  O500=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000_at500__w15__ext.pt O1000=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000_at1000__w15__ext.pt O2000=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000__w15__ext.pt --pairs M16:O500 M16:O2000 O500:O2000
echo "$(date '+%F %T') LANE124_DONE" >> $LOGDIR/lanes.log
