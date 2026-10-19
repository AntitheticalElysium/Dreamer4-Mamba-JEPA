source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 12:52: can moved content be found without labels? (feasibility of a generic E22Z mask; labels validate only)
job e22_motionmask 1100 $PY $L/e22_motionmask.py artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE154_DONE" >> $LOGDIR/lanes.log
