source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 10:39: E22F smoke (predeclared 10:37, NOTEBOOK): Park & Lee phase 1 on D (D and the Delta projection frozen), 2,000 updates,
# batch 16 L16; then P1 / P2 (health_zombie) and the channel diagnostics (e22_channel).
job e22f_smoke 2200 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --updates 2000 --windows 16 --freeze-world
job e22f_p2 1500 $PY $L/health_zombie.py E22F=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt
job e22f_channel 1500 $PY $L/e22_channel.py E22F=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt --out e22_channel_e22f.json
echo "$(date '+%F %T') LANE147_DONE" >> $LOGDIR/lanes.log
