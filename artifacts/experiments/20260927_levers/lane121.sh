source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 21:50: the rest of E21 after GE, re-plumbed (NOTEBOOK "E' = LP-FT"; replaces the stopped lanes 110 / 114 / 115).
#  - LP step for E' runs in lane120 (in parallel; E' waits for its checkpoint);
#  - the D patch lands BETWEEN trainings (after GE's training process exits): tworld.py stamps script_sha256 from the file at
#    save time, so patching under a running training would mislabel its checkpoints; then the CPU equivalence check;
#  - lane108 is stopped right after GE's arm (before it starts the random-head E), lane109 after GE's 4k extraction;
#  - E' (LP-FT) and D train; every readout of the stopped lanes runs on C0 / G / GE / E' / D.
W=artifacts/eda/levers_tworlds_v1; E=artifacts/eda/health_chain_v1
M16=corrt_rawlong_teacher_s7_fmamba_L16b40_from36000
TAIL=_L16b40_from$M16
C0=corrt_rawlong_teacher_s7_fmamba$TAIL; G=corrt_rawlong_teacher_s7_fmamba_gl$TAIL
GE=corrt_rawlong_teacher_s7_fmamba_gl_ev$TAIL; D=corrt_rawlong_teacher_s7_fmamba_di$TAIL
EVLP=corrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000
EP=corrt_rawlong_teacher_s7_fmamba_ev_L16b40_from$EVLP
until grep -q "DONE e21v2_train_gl_ev" $LOGDIR/lanes.log; do sleep 5; done
$PY $L/e21_patch_d.py $L/tworld.py && echo "$(date '+%F %T') PATCHED tworld.py (E21 arm D, --recon) after GE's training exited" >> $LOGDIR/lanes.log
job e21_equivalence_patched 0 env CUDA_VISIBLE_DEVICES= $PY $L/tworld_equivalence.py
until grep -q "E21V2_ARM_DONE gl_ev" $LOGDIR/lanes.log; do sleep 2; done
pkill -f "bash artifacts/experiments/20260927_levers/lane108.sh"
sleep 3
if grep -q "START e21v2_train_ev" $LOGDIR/lanes.log; then
  pkill -f -- "--updates 6000 --backbone fmamba --frames 16 --windows 40 --init $W/$M16.pt --snapshots --snapshot-every 2000 --state-every 1000 --event"
  echo "$(date '+%F %T') STOP e21v2_train_ev (random-head E replaced by E')" >> $LOGDIR/lanes.log
fi
echo "$(date '+%F %T') STOP lane108 after GE (E replaced by E' = LP-FT)" >> $LOGDIR/lanes.log
until grep -q "DONE e21v2_hext_gl_ev_at4000\|FAILED e21v2_hext_gl_ev_at4000" $LOGDIR/lanes.log; do sleep 10; done
pkill -f "bash artifacts/experiments/20260927_levers/lane109.sh"; echo "$(date '+%F %T') STOP lane109 after GE's 4k extraction" >> $LOGDIR/lanes.log
until grep -q "LANE120_DONE" $LOGDIR/lanes.log && [ -f $W/$EVLP.pt ]; do sleep 20; done
job e21v2_train_evlp 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
    --frames 16 --windows 40 --init $W/$EVLP.pt --snapshots --snapshot-every 2000 --state-every 1000 --event
job e21v2_train_di 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
    --frames 16 --windows 40 --init $W/$M16.pt --snapshots --snapshot-every 2000 --state-every 1000 --recon deltairis
for N in $EP ${EP}_at2000 ${EP}_at4000 $D ${D}_at2000 ${D}_at4000; do job hext_$N 1100 $PY $L/health_chain.py world $W/$N.pt --ext; done
for N in $C0 $GE $EP $D; do job hext_w10_$N 1100 $PY $L/health_chain.py world $W/$N.pt --window 10 --ext; done
for N in $EP $D; do job teval_$N 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py $W/$N.pt --window 15; done
ext() { for n in "$@"; do echo $E/${n}__w15__ext.pt; done; }
job e21v2_evidence_cd 1100 $PY $L/health_evidence.py e21v2_cd $(ext $GE ${GE}_at2000 ${GE}_at4000 $EP ${EP}_at2000 ${EP}_at4000 $D ${D}_at2000 ${D}_at4000)
job e21v2_evidence_row9 1100 $PY $L/health_evidence.py e21v2_row9 $(for n in $M16 $C0 $G $GE $EP $D; do echo $E/${n}__w10__ext.pt; done)
job e21v2_damage_w15 3000 $PY $L/check_damage.py $W/$M16.pt $W/$C0.pt $W/$G.pt $W/$GE.pt $W/$EP.pt $W/$D.pt --window 15
for N in $C0 $G $GE $EP; do job grad_$N 1500 $PY $L/health_gradient.py $W/$N.pt 20; done
job grad_$D 1500 $PY $L/health_gradient.py $W/$D.pt 20 deltairis
job e21v2_compare 0 $PY $L/compare.py ${M16}__w15:${C0}__w15 ${C0}__w15:${G}__w15 ${C0}__w15:${GE}__w15 ${C0}__w15:${EP}__w15 ${C0}__w15:${D}__w15
echo "$(date '+%F %T') LANE121_DONE" >> $LOGDIR/lanes.log
