source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 night: oracle positive control (health_plasticity.py oracle): can M16 s7's backbone still put the cooldown phase and
# approach adjacency into h63 when asked directly? 2,000 updates, snapshots 500 / 1,000; then extended states and evidence.
O=corrt_rawlong_teacher_s7_fmamba_oracle_L16b40_from36000
job hplast_oracle 3000 $PY $L/health_plasticity.py oracle artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --updates 2000
for N in ${O}_at500 ${O}_at1000 $O; do job hext_$N 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$N.pt --ext; done
job hevid_oracle 1100 $PY $L/health_evidence.py oracle artifacts/eda/health_chain_v1/${O}_at500__w15__ext.pt artifacts/eda/health_chain_v1/${O}_at1000__w15__ext.pt artifacts/eda/health_chain_v1/${O}__w15__ext.pt
echo "$(date '+%F %T') LANE118_DONE" >> $LOGDIR/lanes.log
