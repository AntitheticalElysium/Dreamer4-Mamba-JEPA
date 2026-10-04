source artifacts/experiments/20260927_levers/lib.sh
# E8 (predeclared 2026-10-01): self-feeding recipes that expose every input slot (corrt, Raw, 6k, seed 7)
W=artifacts/eda/levers_tworlds_v1
for loss in noise selffed; do
  N=corrt_raw_${loss}_s7
  [ -f $W/$N.pt ] || job tworld_$N 3000 $PY $L/tworld.py --head corrt --pool raw --loss $loss
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  [ -f $L/evals/${N}__w4_per_root.pt ] || job teval_${N}_w4 2000 $PY $L/teval.py $W/$N.pt --window 4
  [ -f $L/posprofile_$N.json ] || job posprofile_$N 1500 $PY $L/posprofile.py $W/$N.pt
  [ -f $L/blockwin_$N.json ] || job blockwin_$N 1500 $PY $L/blockwin.py $W/$N.pt
done
until grep -q "LANE18A_DONE" $LOGDIR/lanes.log; do sleep 60; done
T=corrt_raw_teacher_s7_u18000_at6000
job compare_e8 0 $PY $L/compare.py $T:corrt_raw_noise_s7 $T:corrt_raw_selffed_s7 corrt_raw_suffix_s7:corrt_raw_selffed_s7 \
  corrt_raw_suffix_s8:corrt_raw_teacher_s8 corrt_raw_suffix_s7:corrt_raw_suffix_s8 $T:corrt_raw_teacher_s8 \
  ${T}__w4:corrt_raw_noise_s7__w4 ${T}__w4:corrt_raw_selffed_s7__w4
echo "$(date '+%F %T') LANE18B_DONE" >> $LOGDIR/lanes.log
