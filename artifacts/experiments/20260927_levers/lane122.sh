source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 night: from-scratch replication of E21 GE (predeclared 21:20, NOTEBOOK "Could a negative come from continuing"):
# the canonical recipe from step 0 with --gen-loss --event. Stage 1 = M6's exact command + flags (36k, 6-frame, snapshots every
# 6k = M6's); stage 2 = M16's exact command + flags (6k, L16, --init stage 1). Matched controls: M6 snapshots, M16.
# (re-issue of lane119, stopped 21:40, now gated on lane121.) Starts after every E21 readout lane (their 3 GB jobs would otherwise wait behind a 7 h training); first re-checks that tworld.py
# with flags off still trains exactly as M6's / M16's sources (tworld_equivalence.py, CPU).
S1=corrt_raw_teacher_s7_fmamba_gl_ev_u36000
S2=corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000
for D in LANE121_DONE LANE118_DONE; do until grep -q "$D" $LOGDIR/lanes.log; do sleep 60; done; done
job scratch_equivalence 0 env CUDA_VISIBLE_DEVICES= $PY $L/tworld_equivalence.py
job scratch_ge_stage1 3000 $PY $L/tworld.py --head corrt --pool raw --loss teacher --seed 7 --updates 36000 --backbone fmamba \
    --snapshots --state-every 2000 --gen-loss --event
job scratch_ge_stage2 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
    --frames 16 --windows 40 --init artifacts/eda/levers_tworlds_v1/$S1.pt --state-every 1000 --gen-loss --event
for U in 6000 12000 18000 24000 30000; do job hext_${S1}_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/${S1}_at$U.pt --ext; done
job hext_$S1 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S1.pt --ext
job hext_$S2 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S2.pt --ext
job hext_w10_$S2 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S2.pt --window 10 --ext
job teval_$S2 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py artifacts/eda/levers_tworlds_v1/$S2.pt --window 15
job hevid_scratch_ge 1100 $PY $L/health_evidence.py scratch_ge artifacts/eda/health_chain_v1/${S1}_at6000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at12000__w5__ext.pt \
    artifacts/eda/health_chain_v1/${S1}_at18000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at24000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at30000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}__w5__ext.pt \
    artifacts/eda/health_chain_v1/${S2}__w15__ext.pt artifacts/eda/health_chain_v1/${S2}__w10__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt
job damage_$S2 3000 $PY $L/check_damage.py artifacts/eda/levers_tworlds_v1/$S2.pt --window 15
job grad_$S2 1500 $PY $L/health_gradient.py artifacts/eda/levers_tworlds_v1/$S2.pt 20
job compare_scratch_ge 0 $PY $L/compare.py corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15:${S2}__w15
echo "$(date '+%F %T') LANE122_DONE" >> $LOGDIR/lanes.log
