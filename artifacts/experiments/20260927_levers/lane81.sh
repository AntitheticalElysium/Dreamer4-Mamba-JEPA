#!/usr/bin/env bash
# Lead queue: finish the health replication, then registration before further controls/long readouts.
# Oct6 reboot reorder changes scheduling only; all predeclared treatments and budgets are retained.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
W=artifacts/eda/levers_tworlds_v1
$PY -B - <<'PY'
import json
import hashlib
from pathlib import Path
p=Path('artifacts/experiments/20260927_levers')
for device in ('cpu','cuda'):
 v=json.loads((p/f'e19_verify_{device}.json').read_text())
 assert v['A_loss_difference']==0 and v['A_gradient_max_abs']<=1e-6
 assert v['B_C_teacher_difference']<=1e-6 and v['future_perturbation_max_abs']<=1e-5
 assert v['changed_contract_rejected'] and v['corrupt_payload_rejected']
 for path, digest in v['source'].items():
  assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest, f'Code changed since verification: {path}'
v=json.loads((p/'e19_verify_cuda.json').read_text())
for arm in 'ABC':
 m=v['four_updates_vs_two_plus_two'][arm]
 assert m['parameter_max_abs']<=1e-6 and m['optimizer_max_abs']<=1e-6
 assert all(m[k] for k in ('order_equal','boundaries_equal','cpu_rng_equal','cuda_rng_equal'))
v=json.loads((p/'e19_batch40_smoke.json').read_text())
assert len(v)==6 and all(m['batch']==40 and m['peak_allocated_gb']<4.5 for m in v.values())
PY

e19_train() {
  local seed=$1 bb=$2 arm=$3
  job e19_${bb}_s${seed}_${arm} 3000 $PY -B $L/e19.py --arm "$arm" --backbone "$bb" --seed "$seed" --updates 6000 --state-every 500
}
e19_set() {
  local seed=$1 bb=$2 parent=corrt_raw_teacher_s${1}_u36000
  [ "$bb" = fmamba ] && parent=corrt_raw_teacher_s${seed}_fmamba_u36000
  for arm in A B C; do e19_train "$seed" "$bb" "$arm"; done
  local worlds="$W/$parent.pt"
  local roots="$L/evals/${parent}__e19_health_per_root.pt"
  for arm in A B C; do
    worlds="$worlds $W/e19_${arm}_s${seed}_${bb}_from36000.pt"
    roots="$roots $L/evals/e19_${arm}_s${seed}_${bb}_from36000__e19_health_per_root.pt"
  done
  job e19_positions_${bb}_s${seed} 3000 $PY -B $L/e19_eval.py $worlds
  job e19_position_contrasts_${bb}_s${seed} 0 $PY -B $L/e19_eval.py --compare $roots
  job e19_damage_${bb}_s${seed} 3000 $PY -B $L/check_damage.py $worlds
  for arm in A B C; do
    local name=e19_${arm}_s${seed}_${bb}_from36000
    job e19_teval_${bb}_s${seed}_${arm} 3000 $PY -B artifacts/experiments/20261005_recovery/teval_export.py "$W/$name.pt"
    job e19_compare_${bb}_s${seed}_${arm} 0 $PY -B $L/compare.py "$parent:$name"
  done
  $PY -B $L/e19_note.py --seed "$seed" --backbone "$bb"
}

# Recover the original M16 continuation if necessary. Completed seed7 worlds are
# checked against their hash-bound report/raw-row records without replaying every export.
N=corrt_rawlong_teacher_s8_fmamba_L16b40_from36000
if [ ! -f "$W/$N.pt" ]; then
  job e17_M16_s8 3200 $PY -B $L/tworld.py --head corrt --loss teacher --seed 8 --backbone fmamba --pool rawlong \
    --frames 16 --windows 40 --updates 6000 --state-every 1000 --init "$W/corrt_raw_teacher_s8_fmamba_u36000.pt" \
    --resume "$W/state/$N.state.pt"
fi
if [ -f "$W/state/e19_C_s7_fmamba_from36000/complete.json" ] && \
   [ -f "$L/evals/e19_C_s7_fmamba_from36000__readout_v2.json" ]; then
  job e19_verified_completed_s7 0 $PY -B artifacts/experiments/20261005_recovery/teval_export.py \
    "$W/e19_A_s7_fmamba_from36000.pt" "$W/e19_B_s7_fmamba_from36000.pt" "$W/e19_C_s7_fmamba_from36000.pt"
else
  e19_set 7 fmamba
fi

# Cheap E17 health/recall readings precede replicas and the long heads. Mathematical treatments unchanged.
L16=""
for s in 7 8; do L16="$L16 $W/corrt_rawlong_teacher_s${s}_L16b40_from36000.pt $W/corrt_rawlong_teacher_s${s}_fmamba_L16b40_from36000.pt"; done
if [ ! -f "$L/evals/e17_fixed_clock_contrasts.json" ] || \
   ! grep -q 'DONE e17_recall_imagined_w15' "$LOGDIR/lanes.log"; then
  job e17_damage_w15 3000 $PY -B $L/check_damage.py $L16 --window 15
  job e17_damage_w5 3000 $PY -B $L/check_damage.py $L16
  job e17_recall_w15 3000 $PY -B $L/check_recall.py --futures $L16
  job e17_recall_w5 3000 $PY -B $L/check_recall.py --futures --window 5 $L16
  job e17_recall_imagined_w15 3000 $PY -B $L/check_recall.py --futures --imagined $L16
fi
e19_set 8 fmamba

# Preserve E18's already-declared model and state; coordinate registration is a separate intervention.
# Fixed-clock E17 now resolves same-slot history use; prioritize its registration contrast over
# more replicas of the failed seed7 health treatment. Do not silently promote or drop any E19 arm.
/bin/bash "$L/lane74.sh" || exit $?

e19_set 7 full
e19_set 8 full

# Complete original E17 endpoint suite after the mechanism/training lanes; completed batches/heads reuse their journals.
/bin/bash "$L/lane73.sh" || exit $?
/bin/bash artifacts/experiments/20261005_recovery/h16_seed8.sh || exit $?
echo "$(date '+%F %T') LANE81_DONE" >> "$LOGDIR/lanes.log"
