LOG=artifacts/eda/levers_logs/lanes.log
until grep -q "START tworld_direct_raw_suffix_s8" $LOG || grep -q "FAILED tworld_residual_tc_suffix_s8" $LOG; do sleep 5; done
systemctl --user stop lev-lane1
echo "$(date '+%F %T') DEFER lane1 direct_raw_s8 behind lanes 9/10 (GPU memory); lane1b takes its tail" >> $LOG
systemd-run --user --unit=lev-lane1b --collect -p MemoryHigh=10G -p MemoryMax=12G --working-directory=$PWD bash artifacts/experiments/20260927_levers/lane1b.sh
