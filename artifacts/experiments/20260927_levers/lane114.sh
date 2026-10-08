source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 E21 arm D (predeclared 17:50): once E21's last arm (E) has started training, apply e21_patch_d.py to tworld.py
# (--recon deltairis; default unchanged), then, after lane108, train D (6,000 updates from M16 s7, Delta-IRIS tokenizer loss in token
# space) with snapshots, extract states, teval, and read it out like the other arms.
W=artifacts/eda/levers_tworlds_v1; E=artifacts/eda/health_chain_v1
M16=corrt_rawlong_teacher_s7_fmamba_L16b40_from36000
TAIL=_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000
C0=corrt_rawlong_teacher_s7_fmamba$TAIL; D=corrt_rawlong_teacher_s7_fmamba_di$TAIL
until grep -q "START e21v2_train_ev" $LOGDIR/lanes.log; do sleep 30; done
sleep 180                                                          # E's process has imported tworld.py
$PY $L/e21_patch_d.py $L/tworld.py && echo "$(date '+%F %T') PATCHED tworld.py (E21 arm D, --recon)" >> $LOGDIR/lanes.log
until grep -q "LANE108_DONE" $LOGDIR/lanes.log; do sleep 30; done
job e21v2_train_di 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
    --frames 16 --windows 40 --init $W/$M16.pt --snapshots --snapshot-every 2000 --state-every 1000 --recon deltairis
job e21v2_hext_di 3000 $PY $L/health_chain.py world $W/$D.pt --ext
for U in 2000 4000; do job e21v2_hext_di_at$U 1100 $PY $L/health_chain.py world $W/${D}_at$U.pt --ext; done
job e21v2_teval_di 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py $W/$D.pt --window 15
until grep -q "LANE110_DONE" $LOGDIR/lanes.log; do sleep 30; done
job e21v2_evidence_d 1100 $PY $L/health_evidence.py e21v2_d $E/${D}__w15__ext.pt $E/${D}_at2000__w15__ext.pt $E/${D}_at4000__w15__ext.pt
job e21v2_damage_di 3000 $PY $L/check_damage.py $W/$D.pt --window 15
job e21v2_grad_di 1500 $PY $L/health_gradient.py $W/$D.pt 20 deltairis
job e21v2_compare_di 0 $PY $L/compare.py ${C0}__w15:${D}__w15
echo "$(date '+%F %T') LANE114_DONE" >> $LOGDIR/lanes.log
