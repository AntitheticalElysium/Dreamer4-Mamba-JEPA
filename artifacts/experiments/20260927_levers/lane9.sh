source artifacts/experiments/20260927_levers/lib.sh
# Stage 2: per-tile backbones (tworld --backbone), corrt head, Raw tokens, suffix loss, seed 7; one switch each
W=artifacts/eda/levers_tworlds_v1
for bb in fmamba fcanvas fattn fscan; do
  name=corrt_raw_suffix_s7_$bb
  [ -f $W/$name.pt ] || job tworld_$name 3600 $PY $L/tworld.py --head corrt --pool raw --loss suffix --backbone $bb
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_stage2 0 $PY $L/compare.py corrt_raw_suffix_s7:corrt_raw_suffix_s7_fattn corrt_raw_suffix_s7_fattn:corrt_raw_suffix_s7_fmamba \
  corrt_raw_suffix_s7_fmamba:corrt_raw_suffix_s7_fcanvas corrt_raw_suffix_s7_fmamba:corrt_raw_suffix_s7_fscan corrt_raw_suffix_s7:corrt_raw_suffix_s7_fcanvas
echo "$(date '+%F %T') LANE9_DONE" >> $LOGDIR/lanes.log
