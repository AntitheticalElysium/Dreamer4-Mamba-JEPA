#!/usr/bin/env bash
# Lane 3d (E5e): is the corrg world's weak move decision (gate AUC 0.79 while the target tile's own output holds
# passability at 1.00) under-training, or routing? corrt reads the target tile directly; corrg at 3x updates.
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
arm() { local name=$1 need=$2; shift 2; [ -f $W/$name.pt ] || job tworld_$name $need $PY $L/tworld.py "$@"; }
arm corrt_raw_suffix_s7        3600 --head corrt --pool raw --loss suffix
arm corrg_raw_suffix_s7_u18000 3600 --head corrg --pool raw --loss suffix --updates 18000
for name in corrt_raw_suffix_s7 corrg_raw_suffix_s7_u18000; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_corrt 0 $PY $L/compare.py corrg_raw_suffix_s7:corrt_raw_suffix_s7 corrg_raw_suffix_s7:corrg_raw_suffix_s7_u18000 residual_raw_suffix_s7:corrt_raw_suffix_s7
echo "$(date '+%F %T') LANE3D_DONE" >> $LOGDIR/lanes.log
