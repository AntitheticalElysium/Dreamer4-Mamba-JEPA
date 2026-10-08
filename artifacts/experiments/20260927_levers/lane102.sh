source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 health diagnosis 6: existing checkpoints that bear on the two blockers. B2 (generator): the ITC-derived generator
# loss (--gen-loss, the generator's own L1 on every token) vs its matched suffix baselines at 6k / 18k. B1 + B2 under a health
# dose: E19 B (no dose) vs C (dose) from the same M6 36k parent.
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE99_DONE" $LOGDIR/lanes.log; do sleep 20; done
for N in corrt_raw_suffix_s7 corrt_raw_suffix_s7_gl corrt_raw_suffix_s7_u18000 corrt_raw_suffix_s7_gl_itc_u18000 \
         e19_B_s7_fmamba_from36000 e19_C_s7_fmamba_from36000; do
  job hext_$N 3000 $PY $L/health_chain.py world $W/$N.pt --ext
done
echo "$(date '+%F %T') LANE102_DONE" >> $LOGDIR/lanes.log
