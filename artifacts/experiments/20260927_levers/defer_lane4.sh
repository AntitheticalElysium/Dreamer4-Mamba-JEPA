# stop lane 4 as soon as it starts raw lambda 1, and hand its tail to lane 4b
LOG=artifacts/eda/levers_logs/lanes.log
until grep -q "START ldad_raw_lam1_full" $LOG || grep -q "FAILED ldad_tc_lam10_full" $LOG; do sleep 5; done
systemctl --user stop lev-lane4
sleep 5; rm -rf artifacts/eda/levers_ldad_v1/raw_lam1
echo "$(date '+%F %T') DEFER lane4 raw_lam1 behind lanes 9/10 (GPU memory); lane4b takes its tail" >> $LOG
systemd-run --user --unit=lev-lane4b --collect -p MemoryHigh=10G -p MemoryMax=12G --working-directory=$PWD bash artifacts/experiments/20260927_levers/lane4b.sh
