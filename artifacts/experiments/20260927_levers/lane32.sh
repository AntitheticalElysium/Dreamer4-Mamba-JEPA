source artifacts/experiments/20260927_levers/lib.sh
# E14a addendum (2026-10-02; arms and readings in headfit.py's docstring, committed before this lane ran): SimPLe dead zone,
# mask dose response, MLP readout on the same frozen backbone (capacity or representation?)
W=artifacts/eda/levers_tworlds_v1
until grep -q "LANE30_DONE" $LOGDIR/lanes.log; do sleep 60; done
CONT=clip50,clip75,clip90,mask0.1,mask1,mask3,mlp_uniform,mlp_mask10,mlp_mask1
for w in corrt_raw_teacher_s7_u18000 corrt_raw_teacher_s8_u18000 direct_raw_suffix_s7; do
  grep -q '"mlp_mask1"' $L/evals/headfit_$w.json 2>/dev/null || job headfit_add_$w 2600 $PY $L/headfit.py --arms $CONT $W/$w.pt
done
grep -q '"mlp_mask1"' $L/evals/headfit_categorical_raw_teacher_s7_K4096.json 2>/dev/null || job headfit_add_categorical_raw_teacher_s7_K4096 2600 \
  $PY $L/headfit.py --arms $CONT,ce_clip03 $W/categorical_raw_teacher_s7_K4096.pt
# E14b budget control (declared here before it ran): corrt teacher at 36,000 updates, seeds 7 and 8, snapshots every 6,000
# (constant LR after warmup: the first 18,000 updates replay the 18k worlds). consfit on every snapshot. Readings (held strict
# caught): budget_fixes = >= 0.5 at 36k at BOTH seeds; budget_none_s7 = s7 < 0.05 at 36k; else partial (trajectory reported);
# reproducible = the at18000 snapshot's held caught within 0.02 of the existing u18000 world's.
for s in 7 8; do
  [ -f $W/corrt_raw_teacher_s${s}_u36000.pt ] || job tworld_corrt_raw_teacher_s${s}_u36000 3100 $PY $L/tworld.py --head corrt \
    --loss teacher --seed $s --updates 36000 --snapshots
done
job consfit_36k 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u36000_at18000.pt $W/corrt_raw_teacher_s7_u36000_at24000.pt \
  $W/corrt_raw_teacher_s7_u36000_at30000.pt $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u36000_at18000.pt \
  $W/corrt_raw_teacher_s8_u36000_at24000.pt $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_u36000.pt
echo "$(date '+%F %T') LANE32_DONE" >> $LOGDIR/lanes.log
