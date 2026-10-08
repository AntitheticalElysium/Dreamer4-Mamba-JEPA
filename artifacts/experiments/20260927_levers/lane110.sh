source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 v2 readouts. Phase 1, once G and C0 exist: subset evidence for M16, G, C0 and their 2k / 4k snapshots (alongside
# GE's training; probes ~1 GB). Phase 2, after every arm: GE and E evidence, the simulator-truth damage panel at window 15
# (check_damage.py), per-term gradient allocation (health_gradient.py), teval comparisons vs C0.
W=artifacts/eda/levers_tworlds_v1; E=artifacts/eda/health_chain_v1
M16=corrt_rawlong_teacher_s7_fmamba_L16b40_from36000
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
C0=corrt_rawlong_teacher_s7_fmamba$TAIL; GE=corrt_rawlong_teacher_s7_fmamba_gl_ev$TAIL
EV=corrt_rawlong_teacher_s7_fmamba_ev$TAIL; GL=corrt_rawlong_teacher_s7_fmamba_gl$TAIL
ext() { for n in "$@"; do echo $E/${n}__w15__ext.pt; done; }
until grep -q "E21V2_ARM_DONE c0" $LOGDIR/lanes.log; do sleep 30; done
until [ -f $E/${C0}_at4000__w15__ext.pt ]; do sleep 30; done
job e21v2_evidence_ab 1100 $PY $L/health_evidence.py e21v2_ab $(ext $M16 $GL ${GL}_at2000 ${GL}_at4000 $C0 ${C0}_at2000 ${C0}_at4000)
until grep -q "LANE108_DONE" $LOGDIR/lanes.log; do sleep 30; done
until grep -q "LANE109_DONE" $LOGDIR/lanes.log; do sleep 30; done
job e21v2_evidence_cd 1100 $PY $L/health_evidence.py e21v2_cd $(ext $GE ${GE}_at2000 ${GE}_at4000 $EV ${EV}_at2000 ${EV}_at4000)
job e21v2_damage_w15 3000 $PY $L/check_damage.py $W/$M16.pt $W/$C0.pt $W/$GL.pt $W/$GE.pt $W/$EV.pt --window 15
for N in $C0 $GL $GE $EV; do
  job e21v2_grad_${N:0:40} 3000 $PY $L/health_gradient.py $W/$N.pt 20
done
job e21v2_compare 0 $PY $L/compare.py ${M16}__w15:${C0}__w15 ${C0}__w15:${GL}__w15 ${C0}__w15:${GE}__w15 ${C0}__w15:${EV}__w15
echo "$(date '+%F %T') LANE110_DONE" >> $LOGDIR/lanes.log
