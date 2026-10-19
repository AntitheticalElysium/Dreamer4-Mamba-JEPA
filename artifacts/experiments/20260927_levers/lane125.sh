source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 23:05: sufficiency test of the diagnosis (diagnostic, labels): M16 + oracle phase/adjacency (B1) + generator loss (B2),
# 2,000 updates; read at row 14 and row 9 beside the oracle-only world and M16.
until grep -q "DONE e21v2_train_evlp\|FAILED e21v2_train_evlp" $LOGDIR/lanes.log; do sleep 30; done
job hplast_oracle_gl 3000 $PY $L/health_plasticity.py oracle artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --updates 2000 --gen-loss
job hext_corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000.pt --ext
job hext_w10_corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000.pt --window 10 --ext
until grep -q "LANE124_DONE" $LOGDIR/lanes.log; do sleep 30; done
job hevid_sufficiency 1100 $PY $L/health_evidence.py sufficiency artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000__w15__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_gl_L16b40_from36000__w10__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000__w10__ext.pt
echo "$(date '+%F %T') LANE125_DONE" >> $LOGDIR/lanes.log
