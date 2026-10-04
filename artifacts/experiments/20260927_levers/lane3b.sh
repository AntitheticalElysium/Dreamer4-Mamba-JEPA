#!/usr/bin/env bash
# Lane 3b (E5b): the corr head learned the scroll (moves 0.91 -> 0.20 x copy) but scrolls on blocked moves too
# (6.1 x copy). Arms: corrg (frame-level move gate from the action token), corr seed 8, corr on TC tokens.
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
arm() { local name=$1 need=$2; shift 2; [ -f $W/$name.pt ] || job tworld_$name $need $PY $L/tworld.py "$@"; }
arm corrg_raw_suffix_s7 3600 --head corrg --pool raw --loss suffix
arm corr_raw_suffix_s8  3600 --head corr --pool raw --loss suffix --seed 8
arm corr_tc_suffix_s7   3600 --head corr --pool tc --loss suffix
for name in corrg_raw_suffix_s7 corr_raw_suffix_s8 corr_tc_suffix_s7; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_corr 0 $PY $L/compare.py corr_raw_suffix_s7:corrg_raw_suffix_s7 corr_raw_suffix_s7:corr_raw_suffix_s8 corr_raw_suffix_s7:corr_tc_suffix_s7
echo "$(date '+%F %T') LANE3B_DONE" >> $LOGDIR/lanes.log
