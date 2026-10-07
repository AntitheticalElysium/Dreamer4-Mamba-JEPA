source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 4: extended hidden states for (a) the L16 continuation's start (u500) and E20 A, (b) a head
# comparison at matched training: residual vs corrt attention teacher worlds at 6k updates (+ direct, suffix loss).
W=artifacts/eda/levers_tworlds_v1
for N in corrt_rawlong_teacher_s7_fmamba_L16b40_from36000_u500 e20_A_s7_fmamba_fromM16 corrt_raw_teacher_s7_u18000_at6000 \
         residual_raw_teacher_s7 direct_raw_suffix_s7 corrt_raw_teacher_s8_fmamba_u36000; do
  job hext_$N 3000 $PY $L/health_chain.py world $W/$N.pt --ext
done
echo "$(date '+%F %T') LANE99_DONE" >> $LOGDIR/lanes.log
