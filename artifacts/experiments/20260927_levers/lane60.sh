source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: the s7 Mamba world's evaluations ahead of lane52's end (same jobs as lane52's evaluation loop; lane52 repeats the
# cheap ones for both seeds). Readings as declared in lane52.
W=artifacts/eda/levers_tworlds_v1
N=corrt_raw_teacher_s7_fmamba_u36000; A=corrt_raw_teacher_s7_u36000
job consfit_m6_s7 2000 $PY $L/consfit.py $W/$N.pt
job check_damage_m6_s7 1300 $PY $L/check_damage.py $W/$N.pt
[ -f $L/evals/${N}_per_root.pt ] || job teval_m6_s7 2000 $PY $L/teval.py $W/$N.pt
job compare_m6_s7 0 $PY $L/compare.py $A:$N
echo "$(date '+%F %T') LANE60_DONE" >> $LOGDIR/lanes.log
