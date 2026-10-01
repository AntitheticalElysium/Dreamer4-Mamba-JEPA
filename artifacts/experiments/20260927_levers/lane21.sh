source artifacts/experiments/20260927_levers/lib.sh
# E8 diagnostic (2026-10-01): blocked-move scroll rate of the noise world vs the noise level it is told (frames clean)
[ -f $L/blockwin_corrt_raw_noise_s7_level9.json ] || job blocklevel_noise 1500 $PY $L/blocklevel.py artifacts/eda/levers_tworlds_v1/corrt_raw_noise_s7.pt 0 3 6 9
echo "$(date '+%F %T') LANE21_DONE" >> $LOGDIR/lanes.log
