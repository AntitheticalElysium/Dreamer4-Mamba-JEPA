source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E16 diagnosis: M16 s7's health allocation under Delta-IRIS's tokenizer loss (health_gradient.py deltairis), same batches.
W=artifacts/eda/levers_tworlds_v1
job hgrad_deltairis_m16s7 2000 $PY $L/health_gradient.py $W/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt 20 deltairis
echo "$(date '+%F %T') LANE111_DONE" >> $LOGDIR/lanes.log
