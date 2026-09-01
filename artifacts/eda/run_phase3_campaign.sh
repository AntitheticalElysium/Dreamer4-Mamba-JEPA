#!/usr/bin/env bash
# The unattended Phase-3 window: 2.5k actors, a real-fork safety and exploitation check
# that can veto an arm, fresh 10k actors for whichever arms pass, then native-horizon
# paired execution staged 64 seeds then 512. 2.5k and 10k are a declared budget
# comparison -- both are executed and both are reported, whichever wins.
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }

train() {  # arm tag steps
  if [ -f "v2_phase3_$1$2/phase3_final.pt" ]; then say "actor $1$2 already done"; return 0; fi
  say "actor $1$2: $3 steps"
  $PY run_v2_phase3.py --arm "$1" --tag "$2" --steps "$3" || { say "FAILED actor $1$2"; return 1; }
}

for arm in attention mamba; do train "$arm" "" 2500 || exit 1; done

say "safety and exploitation check on real forks"
for arm in attention mamba; do
  $PY check_actor_safety.py --arm "$arm" || { say "FAILED safety $arm"; exit 1; }
done

BUDGETS=("")
for arm in attention mamba; do
  if $PY -c "import json,sys; sys.exit(0 if json.load(open('v2_phase3_$arm/actor_safety.json'))['unsafe'] else 1)"; then
    say "VETO $arm: the 2.5k actor raises true death, holding it at 2.5k for diagnosis"
  else
    train "$arm" "_10k" 10000 || exit 1
    $PY check_actor_safety.py --arm "$arm" --tag _10k || say "safety $arm _10k failed, continuing"
    BUDGETS=("" "_10k")
  fi
done

# Preliminary first: a complete report on 64 paired seeds, then the same block extended
# to 512 reusing every cached episode. If the window runs out, 64 seeds already stand.
for count in 64 512; do
  say "paired native-horizon execution, $count seeds"
  $PY run_paired_execution.py --episodes "$count" --budgets "${BUDGETS[@]}" \
    || { say "FAILED paired execution $count"; exit 1; }
done
say "campaign complete"
