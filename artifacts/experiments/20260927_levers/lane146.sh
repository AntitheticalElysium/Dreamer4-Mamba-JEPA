source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 10:31: the 09:58 rule came out "in between" (cond RMS 6.1x < 10x): its target, Delta's size against the tile embedding, measured.
W=artifacts/eda/levers_tworlds_v1/dworld_a_s7_fromcorrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_revalways_L16_di_b16_u2000.pt
S=artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_revalways_L16_di_b16_u2000.pt
job e22_channel_v2 1500 $PY $L/e22_channel.py WARM=$W SCRATCH=$S --out e22_channel_v2.json
echo "$(date '+%F %T') LANE146_DONE" >> $LOGDIR/lanes.log
