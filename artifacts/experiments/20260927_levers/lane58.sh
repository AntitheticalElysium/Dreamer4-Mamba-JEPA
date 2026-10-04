source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: E16 amendment 2 smoke: stage A from scratch (Delta-IRIS's recipe and Crafter revival), 3,000 updates, logging the
# Delta gain (same-batch teacher loss with Delta zeroed minus with Delta) and codebook usage. Then the matched comparators (lane55).
job dworld_scratch_smoke 2200 $PY $L/dworld.py --stage a --seed 7 --updates 3000
echo "$(date '+%F %T') LANE58_DONE" >> $LOGDIR/lanes.log
systemd-run --user --unit=lev-lane55d --property=MemoryMax=20G --property=MemorySwapMax=0 --working-directory=$PWD /usr/bin/bash artifacts/experiments/20260927_levers/lane55.sh
