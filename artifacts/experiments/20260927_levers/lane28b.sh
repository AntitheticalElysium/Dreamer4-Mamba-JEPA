source artifacts/experiments/20260927_levers/lib.sh
# E13 realign arms (exact-position oracle) for the corrt worlds
W=artifacts/eda/levers_tworlds_v1
job subst16_realign 2000 $PY $L/subst16.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt --arms base,realign,realign+cons,realign+enter,realign+cons+enter --tag realign
echo "$(date '+%F %T') LANE28B_DONE" >> $LOGDIR/lanes.log
