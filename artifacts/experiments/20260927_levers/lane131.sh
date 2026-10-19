source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 02:30: from-scratch replication of E21 D (predeclared 21:06 selection rule: D runs second when its continuation reading
# is positive): the canonical recipe from step 0 with --recon deltairis. Stage 1 = M6's exact command + flag; stage 2 = M16's + flag.
# Matched controls: M6 snapshots, M16. Gated on lane122 and on tworld.py still being the verified file (sha256 2c31cce7eff9ceb58a59d3244180656f5d9b3ea36d7be96a9801ae7e860df935).
S1=corrt_raw_teacher_s7_fmamba_di_u36000
S2=corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000
until grep -q "LANE122_DONE" $LOGDIR/lanes.log; do sleep 60; done
[ "$(sha256sum $L/tworld.py | cut -d' ' -f1)" = "2c31cce7eff9ceb58a59d3244180656f5d9b3ea36d7be96a9801ae7e860df935" ] || { echo "$(date '+%F %T') FAILED lane131: tworld.py changed since verification" >> $LOGDIR/lanes.log; exit 1; }
job scratch_d_stage1 3000 $PY $L/tworld.py --head corrt --pool raw --loss teacher --seed 7 --updates 36000 --backbone fmamba \
    --snapshots --state-every 2000 --recon deltairis
job scratch_d_stage2 3000 $PY $L/tworld.py --head corrt --pool rawlong --loss teacher --seed 7 --updates 6000 --backbone fmamba \
    --frames 16 --windows 40 --init artifacts/eda/levers_tworlds_v1/$S1.pt --state-every 1000 --recon deltairis
for U in 6000 12000 18000 24000 30000; do job hext_${S1}_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/${S1}_at$U.pt --ext; done
job hext_$S1 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S1.pt --ext
job hext_$S2 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S2.pt --ext
job hext_w10_$S2 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/$S2.pt --window 10 --ext
job teval_$S2 3000 $PY artifacts/experiments/20261005_recovery/teval_export.py artifacts/eda/levers_tworlds_v1/$S2.pt --window 15
job hevid_scratch_d 1100 $PY $L/health_evidence.py scratch_d artifacts/eda/health_chain_v1/${S1}_at6000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at12000__w5__ext.pt \
    artifacts/eda/health_chain_v1/${S1}_at18000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at24000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}_at30000__w5__ext.pt artifacts/eda/health_chain_v1/${S1}__w5__ext.pt \
    artifacts/eda/health_chain_v1/${S2}__w15__ext.pt artifacts/eda/health_chain_v1/${S2}__w10__ext.pt
job damage_$S2 3000 $PY $L/check_damage.py artifacts/eda/levers_tworlds_v1/$S2.pt --window 15
job grad_$S2 1500 $PY $L/health_gradient.py artifacts/eda/levers_tworlds_v1/$S2.pt 20 deltairis
job compare_scratch_d 0 $PY $L/compare.py corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15:${S2}__w15
echo "$(date '+%F %T') LANE131_DONE" >> $LOGDIR/lanes.log
