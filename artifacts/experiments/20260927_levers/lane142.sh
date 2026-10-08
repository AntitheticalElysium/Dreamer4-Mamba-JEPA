source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 09:55: E22 pre-flight (predeclared 09:35): stage A smoke (D continued with E16's delta channel, revival always, D's loss,
# L16, batch 16, 2,000 updates) logs the delta gain (P1); then P2: one-step zombie predictions with the posterior's delta.
job e22_smoke 2200 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --updates 2000 --windows 16
A=$(ls -t artifacts/eda/levers_tworlds_v1/dworld_a_s7_*_revalways_L16_di_b16_u2000.pt | head -1)
job e22_p2 1500 $PY $L/health_zombie.py E22A=$A D=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE142_DONE" >> $LOGDIR/lanes.log
