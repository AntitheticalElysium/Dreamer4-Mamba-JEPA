source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: E16 stage A smokes after the crafter-setting collapse (dworld_smoke: 3-4 codes, 1.7 bits, teacher loss unchanged):
# Delta-IRIS's other two revival settings, 1,000 updates each, s7 36k; compared on codebook usage and the teacher loss. Not results.
W=artifacts/eda/levers_tworlds_v1
for r in atari always; do
  job dworld_smoke_$r 2200 $PY $L/dworld.py --stage a --seed 7 --init $W/corrt_raw_teacher_s7_u36000.pt --updates 1000 --revival $r
done
echo "$(date '+%F %T') LANE56_DONE" >> $LOGDIR/lanes.log
systemd-run --user --unit=lev-lane55b --property=MemoryMax=20G --property=MemorySwapMax=0 --working-directory=$PWD /usr/bin/bash artifacts/experiments/20260927_levers/lane55.sh
