#!/usr/bin/env bash
# The five-arm matrix at a second training seed. Arms 1-3 replicate the seed-0 result;
# arms 4-5 are the new comparison, and only their weighting differs.
#
#   lane a   production control -> factual -> counterfactual   (the replication)
#   lane b   full-action -> balanced                           (the new comparison)
#
# Every arm resumes from its own 500-step checkpoint, so re-running this script after an
# interrupt continues where it stopped and skips whatever already finished. Lane b waits
# for lane a to finish caching dev latents, which is the only step that wants the GPU for
# the encoder rather than the world.
set -u
cd "$(dirname "$0")"
SEED=20260732
STEPS=20000
# the project venv, not whatever `python` resolves to: the system interpreter carries
# torch but not jax, and `run_stage_a` reaches the Craftax env through it
PY=../../.venv/bin/python
[ -x "$PY" ] || { echo "no interpreter at $PY"; exit 2; }

finished() {   # $1 report file, $2 key that must read $STEPS
  [ -f "$1" ] && "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['$2'])" "$1" \
      2>/dev/null | grep -qx "$STEPS"
}

arm() {        # $1 out dir, rest: extra flags
  local out=$1; shift
  if finished "$out/training_report.json" steps; then echo "== $out already at $STEPS"; return; fi
  echo "== $out"
  "$PY" train_terminal_arms.py --seed "$SEED" --steps "$STEPS" --out "$out" "$@" || return 1
}

case "${1:?lane a, b or score}" in
a)
  if finished production_1b_s2/done.json steps; then
    echo "== production_1b_s2 already at $STEPS"
  else
    echo "== production_1b_s2"
    "$PY" run_production_1b.py --seed "$SEED" --steps "$STEPS" --out production_1b_s2 || exit 1
  fi
  arm terminal_factual_s2        --arm factual        --terminal-roots 4 --terminal-actions 1  || exit 1
  arm terminal_counterfactual_s2 --arm counterfactual --terminal-roots 4 --terminal-actions 1  || exit 1
  ;;
b)
  # wait for lane a to get its encoder off the GPU, but never wait on a lane that died
  for _ in $(seq 240); do
    [ -f production_1b_s2/done.json ] && break
    grep -q "dev latents cached" production_1b_s2/run.log 2>/dev/null && break
    grep -qE "Traceback|Error" lane_a.log 2>/dev/null && { echo "lane a failed; not starting"; exit 1; }
    sleep 30
  done
  arm terminal_full17_s2   --arm counterfactual --terminal-roots 4 --terminal-actions 17 || exit 1
  arm terminal_balanced_s2 --arm counterfactual --terminal-roots 4 --terminal-actions 17 \
                           --balance-outcomes || exit 1
  ;;
score)
  # every arm at every milestone, skipping readings already on disk, then the two
  # analyses: arms 1-3 are the replication, arms 4-5 the new causal comparison
  read_one() {   # $1 folder, $2 tag, $3 milestone (0 = final)
    local out="death_transfer_$2"; local ck="$1/world.pt"
    if [ "$3" != 0 ]; then out="${out}_$(printf %06d "$3")"; ck="$1/world_$(printf %06d "$3").pt"; fi
    if [ -f "$out.json" ]; then echo "-- $out already read"; return; fi
    [ -f "$ck" ] || { echo "-- $ck not written yet, skipped"; return; }
    "$PY" evaluate_death_transfer.py --folder "$1" --tag "$2" --milestone "$3" \
        > /dev/null || return 1
    echo "-- $out"
  }
  read_one production_1b_s2 production_s2 0 || exit 1
  for a in factual counterfactual full17 balanced; do
    [ -d "terminal_${a}_s2" ] || { echo "-- terminal_${a}_s2 missing, skipped"; continue; }
    for m in 5000 10000 13592 0; do read_one "terminal_${a}_s2" "${a}_s2" "$m" || exit 1; done
  done
  analyse() {    # $1 title, $2 output, rest: the two arms; needs every milestone present
    local title=$1 out=$2; shift 2
    for a in "$@"; do for m in _005000 _010000 _013592 ""; do
      [ -f "death_transfer_$a$m.json" ] || {
        echo; echo "-- $title: death_transfer_$a$m.json missing, not run"; return; }
    done; done
    [ -f death_transfer_production_s2.json ] || {
      echo; echo "-- $title: the control has not been read, not run"; return; }
    echo; echo "########## $title ##########"; echo
    "$PY" analyse_death_paired.py --control production_s2 --arms "$@" --out "$out"
  }
  analyse "replication: arms 1-3" death_paired_s2_replication.json \
          factual_s2 counterfactual_s2
  analyse "new comparison: arms 1, 4, 5" death_paired_s2_comparison.json \
          full17_s2 balanced_s2
  ;;
*) echo "lane must be a, b or score"; exit 2;;
esac
echo "lane $1 complete"
