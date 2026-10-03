source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: E16 stage A diagnostic: does the decoder take information from Delta at all? 6,000 updates with the class-default
# revival (keeps the codebook alive: 8.0 bits in its smoke), logging the same-batch teacher loss with Delta zeroed every 500.
# Then the paused matched-budget comparators (lane55).
W=artifacts/eda/levers_tworlds_v1
job dworld_diag6k 2200 $PY $L/dworld.py --stage a --seed 7 --init $W/corrt_raw_teacher_s7_u36000.pt --updates 6000 --revival always
echo "$(date '+%F %T') LANE57_DONE" >> $LOGDIR/lanes.log
systemd-run --user --unit=lev-lane55c --property=MemoryMax=20G --property=MemorySwapMax=0 --working-directory=$PWD /usr/bin/bash artifacts/experiments/20260927_levers/lane55.sh
