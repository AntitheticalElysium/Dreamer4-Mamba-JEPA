source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 readouts (NOTEBOOK "E21"). Phase 1, once GE and C0 exist: subset evidence for M16, GE, C0 and their 2k / 4k
# snapshots (alongside E's training; probes ~1 GB). Phase 2, after every arm: E and G evidence, the simulator-truth damage panel at
# window 15 (check_damage.py, E17's long_hits reading), per-term gradient allocation (health_gradient.py), teval comparisons vs C0.
W=artifacts/eda/levers_tworlds_v1; E=artifacts/eda/health_chain_v1
M16=corrt_rawlong_teacher_s7_fmamba_L16b40_from36000
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
C0=corrt_rawlong_teacher_s7_fmamba$TAIL; GE=corrt_rawlong_teacher_s7_fmamba_gl_ev$TAIL
EV=corrt_rawlong_teacher_s7_fmamba_ev$TAIL; GL=corrt_rawlong_teacher_s7_fmamba_gl$TAIL
ext() { for n in "$@"; do echo $E/${n}__w15__ext.pt; done; }
until grep -q "E21_ARM_DONE c0" $LOGDIR/lanes.log; do sleep 30; done
until [ -f $E/${C0}_at4000__w15__ext.pt ]; do sleep 30; done
job e21_evidence_ab 1100 $PY $L/health_evidence.py e21_ab $(ext $M16 $GE ${GE}_at2000 ${GE}_at4000 $C0 ${C0}_at2000 ${C0}_at4000)
until grep -q "LANE104_DONE" $LOGDIR/lanes.log; do sleep 30; done
until grep -q "LANE105_DONE" $LOGDIR/lanes.log; do sleep 30; done
job e21_evidence_cd 1100 $PY $L/health_evidence.py e21_cd $(ext $EV ${EV}_at2000 ${EV}_at4000 $GL ${GL}_at2000 ${GL}_at4000)
job e21_damage_w15 3000 $PY $L/check_damage.py $W/$M16.pt $W/$C0.pt $W/$GE.pt $W/$EV.pt $W/$GL.pt --window 15
for N in $C0 $GE $EV $GL; do
  job e21_grad_${N:0:40} 3000 $PY $L/health_gradient.py $W/$N.pt 20
done
job e21_compare 0 $PY $L/compare.py ${M16}__w15:${C0}__w15 ${C0}__w15:${GE}__w15 ${C0}__w15:${EV}__w15 ${C0}__w15:${GL}__w15
echo "$(date '+%F %T') LANE106_DONE" >> $LOGDIR/lanes.log
