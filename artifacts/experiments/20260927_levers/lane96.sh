source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health chain (health_chain.py docstring): frozen worlds on the fixed E20-pool subset, teacher-forced on each world's
# trained context. Canonical Mamba L16 (s7, s8), their 6-frame parents, attention L16 / 6-frame, and the health-trained arms.
W=artifacts/eda/levers_tworlds_v1
for N in corrt_rawlong_teacher_s7_fmamba_L16b40_from36000 corrt_rawlong_teacher_s8_fmamba_L16b40_from36000 \
         corrt_raw_teacher_s7_fmamba_u36000 corrt_raw_teacher_s8_fmamba_u36000 \
         corrt_rawlong_teacher_s7_L16b40_from36000 corrt_rawlong_teacher_s8_L16b40_from36000 corrt_raw_teacher_s7_u36000 \
         e19_C_s7_fmamba_from36000 e20_B_s7_fmamba_fromM16 e20_C_s7_fmamba_fromM16; do
  job hchain_$N 3000 $PY $L/health_chain.py world $W/$N.pt
done
echo "$(date '+%F %T') LANE96_DONE" >> $LOGDIR/lanes.log
