source artifacts/experiments/20260927_levers/lib.sh
# E16 full runs (2026-10-03; design + readings predeclared in NOTEBOOK 7e353eba, amendments 1-2): stage A from scratch (Delta-IRIS's
# recipe, Crafter revival, the deterministic arm's init / batches / 36k), seeds 7 and 8; stage B, the prior, 20,000 updates (declared
# here before any stage-B run: 32 x 21-step sequences = 640 transitions per update); then check_e16 (M = 8) and check_h16_traj --e16.
# The from-scratch smoke (3k): Delta gain +70% at 3k, codebook 110 codes; with Delta zeroed that decoder is worse than the
# deterministic world (0.227 vs ~0.113): Delta also carries deterministic content early; whether the decoder takes it over is
# measured, not assumed (check_e16 reports the Delta gain by token class).
W=artifacts/eda/levers_tworlds_v1
for s in 7 8; do
  A=dworld_a_s${s}_scratch_u36000
  [ -f $W/$A.pt ] || job e16_stageA_s$s 2200 $PY $L/dworld.py --stage a --seed $s --updates 36000
  [ -f $W/${A}_prior_u20000.pt ] || job e16_stageB_s$s 2200 $PY $L/dworld.py --stage b --seed $s --init $W/$A.pt --updates 20000
done
for s in 7 8; do
  P=$W/dworld_a_s${s}_scratch_u36000_prior_u20000.pt
  job e16_check_s$s 2400 $PY $L/check_e16.py $P --samples 8
done
job e16_h16 2600 $PY $L/check_h16_traj.py --e16 $W/dworld_a_s7_scratch_u36000_prior_u20000.pt $W/dworld_a_s8_scratch_u36000_prior_u20000.pt
echo "$(date '+%F %T') LANE59_DONE" >> $LOGDIR/lanes.log
