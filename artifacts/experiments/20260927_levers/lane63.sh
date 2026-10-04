source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: reorder for the user's priority on Mamba: when E16 stage B s7 finishes, stop lane59 (idempotent; its s7 outputs are
# kept), run E16 s7's predeclared evaluation now (check_e16 M = 8); lane61 (check_context) and lane62 (E17 smoke) are queued; then
# relaunch lane59 for seed 8.
W=artifacts/eda/levers_tworlds_v1
until grep -q "DONE e16_stageB_s7" $LOGDIR/lanes.log; do sleep 20; done
systemctl --user stop lev-lane59
echo "$(date '+%F %T') STOP lane59 (reorder: E16 s7 evaluation and Mamba diagnostics first)" >> $LOGDIR/lanes.log
job e16_check_s7 2400 $PY $L/check_e16.py $W/dworld_a_s7_scratch_u36000_prior_u20000.pt --samples 8
until grep -q "LANE61_DONE" $LOGDIR/lanes.log && grep -q "LANE62_DONE" $LOGDIR/lanes.log; do sleep 60; done
systemctl --user reset-failed lev-lane59 2>/dev/null
systemd-run --user --unit=lev-lane59 --property=MemoryMax=20G --property=MemorySwapMax=0 --working-directory=$PWD /usr/bin/bash artifacts/experiments/20260927_levers/lane59.sh
echo "$(date '+%F %T') LANE63_DONE" >> $LOGDIR/lanes.log
