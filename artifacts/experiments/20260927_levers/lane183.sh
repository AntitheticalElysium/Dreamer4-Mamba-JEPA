source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-11 09:4x: relaunch after the overnight reboot (lanes 170 / 171b / 178 died with the machine; last lanes.log entry
# 20:44:08 DONE h16_dscratch). In order of value, one at a time (each H16 needs ~2.7-2.9 GB measured):
#  1. region relevance (check_h16_subst) for D scratch -- the M16 arm finished before the reboot (h16subst_L16.log);
#  2. E22MS H16, resumed from its saved draws (0-5 complete; m6 features partial) -> the M1 / M4 / M16 sample readout;
#  3. D continued H16 (the second D-loss world; continued from M16, not an independent seed).
V=artifacts/eda/levers_tworlds_v1
job h16subst_DS 3000 $PY $L/check_h16_subst.py $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --window 15
job h16_e22ms 3000 $PY -B $L/check_h16_traj.py --e16 $V/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_dscratch_u2000_prior_u20000.pt --window 15
job h16_dcont 3000 $PY -B $L/check_h16_traj.py $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --window 15
echo "$(date '+%F %T') LANE183_DONE" >> $LOGDIR/lanes.log
