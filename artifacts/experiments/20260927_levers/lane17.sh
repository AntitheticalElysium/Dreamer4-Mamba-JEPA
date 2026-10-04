source artifacts/experiments/20260927_levers/lib.sh
# E5k confirmation: is corrt's time-position-4 move bias caused by the depth-2 generated suffix?
# Same recipe as corrt_raw_suffix_s7_u18000 except --loss teacher (no generated suffix); snapshots every 6k.
W=artifacts/eda/levers_tworlds_v1
N=corrt_raw_teacher_s7_u18000
[ -f $W/$N.pt ] || job tworld_$N 3600 $PY $L/tworld.py --head corrt --pool raw --loss teacher --updates 18000 --snapshots
for tag in $N ${N}_at6000; do
  [ -f $L/evals/${tag}_per_root.pt ] || job teval_$tag 2000 $PY $L/teval.py $W/$tag.pt
  [ -f $L/evals/${tag}__w4_per_root.pt ] || job teval_${tag}_w4 2000 $PY $L/teval.py $W/$tag.pt --window 4
  [ -f $L/posprofile_$tag.json ] || job posprofile_$tag 1500 $PY $L/posprofile.py $W/$tag.pt
  [ -f $L/blockwin_$tag.json ] || job blockwin_$tag 1500 $PY $L/blockwin.py $W/$tag.pt
done
job compare_teacher18 0 $PY $L/compare.py corrt_raw_suffix_s7_u18000:corrt_raw_teacher_s7_u18000 corrt_raw_suffix_s7_u18000__w4:corrt_raw_teacher_s7_u18000__w4 corrt_raw_suffix_s7_u18000:corrt_raw_suffix_s7_u18000__w4
echo "$(date '+%F %T') LANE17_DONE" >> $LOGDIR/lanes.log
