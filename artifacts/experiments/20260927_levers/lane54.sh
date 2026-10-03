source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: E16 stage A resource / codebook smoke (1,000 updates from the s7 36k world): s/update, peak memory, codebook entropy
# and codes in use before any full run (the CPU test showed 71 -> 5 codes in 3 high-LR steps). Not a result.
W=artifacts/eda/levers_tworlds_v1
job dworld_smoke 2200 $PY $L/dworld.py --stage a --seed 7 --init $W/corrt_raw_teacher_s7_u36000.pt --updates 1000
echo "$(date '+%F %T') LANE54_DONE" >> $LOGDIR/lanes.log
