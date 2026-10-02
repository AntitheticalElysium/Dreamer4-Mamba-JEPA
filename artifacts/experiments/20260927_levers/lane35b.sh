source artifacts/experiments/20260927_levers/lib.sh
# E14c / E14b seed 8 (readings in lane35.sh and lane32.sh, committed before any of these trained)
W=artifacts/eda/levers_tworlds_v1
train() { N=$1; shift; [ -f $W/$N.pt ] || job tworld_$N 2400 $PY $L/tworld.py --head corrt --loss teacher "$@"; }
train corrt_raw_teacher_s8_mask1_skip_u18000 --seed 8 --updates 18000 --weight mask1 --skip
# 2026-10-02 16:00: the 36k s8 budget run first (E14b s7: the uniform loss learns the consequences by 24k -- the second seed
# decides); then the linear arm after the E14c s8 evaluations
train corrt_raw_teacher_s8_u36000 --seed 8 --updates 36000 --snapshots
until [ -f $L/evals/monotone_corrt_raw_teacher_s8_mask1_skip_u18000.json ]; do sleep 60; done
train corrt_raw_teacher_s8_mask1_u18000 --seed 8 --updates 18000 --weight mask1
until [ -f $W/corrt_raw_teacher_s7_u36000.pt ]; do sleep 120; done
job consfit_36k 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u36000_at18000.pt $W/corrt_raw_teacher_s7_u36000_at24000.pt \
  $W/corrt_raw_teacher_s7_u36000_at30000.pt $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u36000_at18000.pt \
  $W/corrt_raw_teacher_s8_u36000_at24000.pt $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_u36000.pt
echo "$(date '+%F %T') LANE35B_DONE" >> $LOGDIR/lanes.log
