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

finished() {   # $1 report file, $2 key that must read $STEPS
  [ -f "$1" ] && python -c "import json,sys;print(json.load(open(sys.argv[1]))['$2'])" "$1" \
      2>/dev/null | grep -qx "$STEPS"
}

arm() {        # $1 out dir, rest: extra flags
  local out=$1; shift
  if finished "$out/training_report.json" steps; then echo "== $out already at $STEPS"; return; fi
  echo "== $out"
  python train_terminal_arms.py --seed "$SEED" --steps "$STEPS" --out "$out" "$@" || return 1
}

case "${1:?lane a or b}" in
a)
  if finished production_1b_s2/done.json steps; then
    echo "== production_1b_s2 already at $STEPS"
  else
    echo "== production_1b_s2"
    python run_production_1b.py --seed "$SEED" --steps "$STEPS" --out production_1b_s2 || exit 1
  fi
  arm terminal_factual_s2        --arm factual        --terminal-roots 4 --terminal-actions 1  || exit 1
  arm terminal_counterfactual_s2 --arm counterfactual --terminal-roots 4 --terminal-actions 1  || exit 1
  ;;
b)
  while [ ! -f production_1b_s2/done.json ] \
     && ! grep -q "dev latents cached" production_1b_s2/run.log 2>/dev/null; do sleep 30; done
  arm terminal_full17_s2   --arm counterfactual --terminal-roots 4 --terminal-actions 17 || exit 1
  arm terminal_balanced_s2 --arm counterfactual --terminal-roots 4 --terminal-actions 17 \
                           --balance-outcomes || exit 1
  ;;
*) echo "lane must be a or b"; exit 2;;
esac
echo "lane $1 complete"
