#!/usr/bin/env bash
# Lane 1: per-tile world arms (E2, E3b, E5) and their evaluations.
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
CB=artifacts/eda/levers_codebooks_v1/raw_K4096.pt
arm() { # name need args...
  local name=$1 need=$2; shift 2
  [ -f $W/$name.pt ] || job tworld_$name $need $PY $L/tworld.py "$@"
}
arm corr_raw_suffix_s7            3600 --head corr --pool raw --loss suffix
arm residual_raw_teacher_s7       2000 --head residual --pool raw --loss teacher
arm categorical_raw_teacher_s7_K4096 3000 --head categorical --pool raw --loss teacher --codebook $CB
arm direct_tc_suffix_s7           3000 --head direct --pool tc --loss suffix
arm residual_raw_suffix_s8        3000 --head residual --pool raw --loss suffix --seed 8
arm residual_tc_suffix_s8         3000 --head residual --pool tc --loss suffix --seed 8
arm direct_raw_suffix_s8          3000 --head direct --pool raw --loss suffix --seed 8
for name in gated_raw_suffix_s7 corr_raw_suffix_s7 residual_raw_teacher_s7 categorical_raw_teacher_s7_K4096 \
            direct_tc_suffix_s7 residual_raw_suffix_s8 residual_tc_suffix_s8 direct_raw_suffix_s8; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
for name in residual_raw_suffix_s7 residual_raw_teacher_s7 direct_raw_suffix_s7; do
  [ -f $L/evals/${name}__snap_raw_K4096.json ] || job snap_$name 2000 $PY $L/teval.py $W/$name.pt --snap $CB
done
echo "$(date '+%F %T') LANE1_DONE" >> $LOGDIR/lanes.log
