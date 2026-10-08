source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 03:30: from-scratch GE stage-1 snapshots read as they appear (beside training), each against M6 s7 at the same update
# (window 5; M6's ext files exist from lane98): generator fidelity, since 1..4, adjacency, event head, emitted health.
S1=corrt_raw_teacher_s7_fmamba_gl_ev_u36000
for U in 6000 12000 18000 24000 30000; do
  until [ -f artifacts/eda/levers_tworlds_v1/${S1}_at$U.pt ]; do sleep 60; done; sleep 60
  job hext_early_${S1}_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/${S1}_at$U.pt --ext
  M6=artifacts/eda/health_chain_v1/corrt_raw_teacher_s7_fmamba_u36000_at${U}__w5__ext.pt
  job hci_scratch_ge_at$U 800 $PY $L/health_ci.py scratch_ge_at$U M6=$M6 GE0=artifacts/eda/health_chain_v1/${S1}_at${U}__w5__ext.pt --pairs M6:GE0
  job inflight_scratch_ge_at$U 600 $PY $L/health_inflight.py ${S1}_at$U
done
echo "$(date '+%F %T') LANE135_DONE" >> $LOGDIR/lanes.log
