source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 11:15: E22FJ smoke (predeclared 11:14): Park & Lee / SV2P phase 2 = joint training from the E22F smoke (nothing frozen),
# 2,000 updates, batch 16 L16 (batch order restarts: the same windows as phase 1); then P1 / P2 and carries + channel.
job e22fj_smoke 2600 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --init artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_frozen_L16_di_b16_u2000.pt --updates 2000 --windows 16
job e22fj_p2 1500 $PY $L/health_zombie.py E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt
job e22fj_carries 1500 $PY $L/e22_channel.py E22FJ=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_L16_di_b16_u2000.pt --out e22_carries_fj.json
echo "$(date '+%F %T') LANE149_DONE" >> $LOGDIR/lanes.log
