#!/usr/bin/env bash
# Lane 6 (E7): Delta-JEPA on the per-tile world. The LDAD lambda-10 encoder's patch tokens (same windows as the Raw
# pool), then the best per-tile head (corrt) and the residual head on them, evaluated and compared with Raw tokens.
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
P=artifacts/eda/spatial_pool_ldad10_v1
[ -f $P/pool.json ] || job pool_ldad10 1500 $PY $L/tc_pool.py artifacts/eda/levers_ldad_v1/raw_lam10/step-010000.pt $P
arm() { local name=$1 need=$2; shift 2; [ -f $W/$name.pt ] || job tworld_$name $need $PY $L/tworld.py "$@"; }
arm corrt_ldad10_suffix_s7    3600 --head corrt --pool ldad10 --loss suffix
arm residual_ldad10_suffix_s7 3000 --head residual --pool ldad10 --loss suffix
for name in corrt_ldad10_suffix_s7 residual_ldad10_suffix_s7; do
  [ -f $L/evals/${name}_per_root.pt ] || job teval_$name 2000 $PY $L/teval.py $W/$name.pt
done
job compare_ldad_t 0 $PY $L/compare.py corrt_raw_suffix_s7:corrt_ldad10_suffix_s7 residual_raw_suffix_s7:residual_ldad10_suffix_s7
echo "$(date '+%F %T') LANE6_DONE" >> $LOGDIR/lanes.log
