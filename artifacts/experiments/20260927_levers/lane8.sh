source artifacts/experiments/20260927_levers/lib.sh
# E4c: Delta-JEPA as published (no SIGReg in the objective), Raw, lambda 10, paired with canonical Raw and raw_lam10
D=artifacts/eda/levers_ldad_v1
[ -f $D/raw_lam10_nosig/step-010000.pt ] || { rm -rf $D/raw_lam10_nosig; job ldad_raw_lam10_nosig 2000 $PY $L/ldad_joint.py --variant raw --lam 10 --no-sigreg; }
JAX_PLATFORMS=cpu job ldad_eval_nosig 1800 $PY $L/ldad_eval.py
JAX_PLATFORMS=cpu job ldad_facts_nosig 1800 $PY $L/ldad_facts.py
echo "$(date '+%F %T') LANE8_DONE" >> $LOGDIR/lanes.log
