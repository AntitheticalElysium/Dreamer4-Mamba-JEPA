source artifacts/experiments/20260927_levers/lib.sh
# E8 (predeclared 2026-10-01): seed-8 replication of the suffix-vs-teacher result (corrt, Raw, 6k)
W=artifacts/eda/levers_tworlds_v1
for loss in suffix teacher; do
  N=corrt_raw_${loss}_s8
  [ -f $W/$N.pt ] || job tworld_$N 3000 $PY $L/tworld.py --head corrt --pool raw --loss $loss --seed 8
  [ -f $L/evals/${N}_per_root.pt ] || job teval_$N 2000 $PY $L/teval.py $W/$N.pt
  [ -f $L/evals/${N}__w4_per_root.pt ] || job teval_${N}_w4 2000 $PY $L/teval.py $W/$N.pt --window 4
  [ -f $L/posprofile_$N.json ] || job posprofile_$N 1500 $PY $L/posprofile.py $W/$N.pt
  [ -f $L/blockwin_$N.json ] || job blockwin_$N 1500 $PY $L/blockwin.py $W/$N.pt
done
echo "$(date '+%F %T') LANE18A_DONE" >> $LOGDIR/lanes.log
