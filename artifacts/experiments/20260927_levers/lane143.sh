source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 09:58: E22 warm-start substitution (NOTEBOOK "Substitution test of the warm start"): the lane142 smoke with the decoder
# from scratch (no --init); then clean P1 + P2 on it.
job e22s_smoke 2200 $PY $L/dworld.py --stage a --long --recon deltairis --revival always --seed 7 --updates 2000 --windows 16
job e22s_p2 1500 $PY $L/health_zombie.py E22S=artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_revalways_L16_di_b16_u2000.pt
echo "$(date '+%F %T') LANE143_DONE" >> $LOGDIR/lanes.log
