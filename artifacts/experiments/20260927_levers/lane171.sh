source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 09:16: 10-10 PROGRAM Part I: E22MS exports (zeroed, 4 prior samples, posterior codes), then H16 decisions with prior-sampled
# futures (--e16, window 15; the prior's end head has the terminal-position bias: reported, not judged).
V=artifacts/eda/levers_tworlds_v1
job sb_export_e22ms 2300 $PY $L/rollout_export.py --prior E22MS=$V/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000_prior_u20000.pt --samples 4 --postcodes
job h16_e22ms 3000 $PY -B $L/check_h16_traj.py --e16 $V/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000_prior_u20000.pt --window 15
echo "$(date '+%F %T') LANE171_DONE" >> $LOGDIR/lanes.log
