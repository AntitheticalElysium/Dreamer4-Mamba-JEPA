source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 10:15: E22 channel diagnostics (predeclared 09:58) on the warm (lane142) and scratch (lane143) smokes, once the scratch
# smoke's file exists.
W=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_L16_di_b16_u2000.pt
S=artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_revalways_L16_di_b16_u2000.pt
while [ ! -f $S ]; do sleep 60; done
job e22_channel 1600 $PY $L/e22_channel.py WARM=$W SCRATCH=$S
echo "$(date '+%F %T') LANE144_DONE" >> $LOGDIR/lanes.log
