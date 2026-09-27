#!/usr/bin/env bash
# Lane 3: early evaluation of the copy-variant arms as soon as they exist (lane 1 skips evaluated arms).
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
until [ -f $W/corr_raw_suffix_s7.pt ]; do sleep 60; done
for name in gated_raw_suffix_s7 corr_raw_suffix_s7; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_copy 0 $PY $L/compare.py residual_raw_suffix_s7:gated_raw_suffix_s7 residual_raw_suffix_s7:corr_raw_suffix_s7 direct_raw_suffix_s7:corr_raw_suffix_s7
echo "$(date '+%F %T') LANE3_DONE" >> $LOGDIR/lanes.log
