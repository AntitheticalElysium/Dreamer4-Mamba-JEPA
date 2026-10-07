source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 2: extended hidden states (health_chain.py --ext) for the evidence-sufficiency probes.
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE96_DONE" $LOGDIR/lanes.log; do sleep 20; done
for N in corrt_rawlong_teacher_s7_fmamba_L16b40_from36000 corrt_rawlong_teacher_s8_fmamba_L16b40_from36000 \
         corrt_rawlong_teacher_s7_L16b40_from36000 corrt_rawlong_teacher_s8_L16b40_from36000 \
         corrt_raw_teacher_s7_fmamba_u36000 corrt_raw_teacher_s7_u36000 e20_B_s7_fmamba_fromM16; do
  job hext_$N 3000 $PY $L/health_chain.py world $W/$N.pt --ext
done
echo "$(date '+%F %T') LANE97_DONE" >> $LOGDIR/lanes.log
