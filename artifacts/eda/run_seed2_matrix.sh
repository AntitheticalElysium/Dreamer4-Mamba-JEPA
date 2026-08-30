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

case "${1:?lane a, b, c, d, e, score, gate or queue}" in
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
gate)
  # PREREGISTERED, written before balanced's 20k reading existed. Balanced earns a
  # replication only if it beats full-action on the stratum balancing targets AND has
  # genuinely recovered from its own 13,592 value. Both need a paired interval clear of
  # zero; either failing stops all terminal-data arms.
  "$PY" - <<'EOF'
import json, sys
import numpy as np
g = np.random.default_rng(20260826)
def v(t, m=0):
    f = f"death_transfer_{t}" + (f"_{m:06d}" if m else "") + ".json"
    return np.array(json.load(open(f))["per_root_pred"])
try:
    b20, b13, f20 = v("balanced_s2"), v("balanced_s2", 13592), v("full17_s2")
except FileNotFoundError as e:
    sys.exit(f"gate cannot run: {e.filename} missing")
lethal = np.array(json.load(open("death_transfer_production_s2.json"))["per_root_lethal"])
e = np.where(lethal <= 2)[0]
draws = e[g.integers(0, len(e), (10000, len(e)))]
def band(d):
    lo, hi = np.quantile(d[draws].mean(1), [0.025, 0.975]); return d[e].mean(), lo, hi
tests = {"balanced(20k) - full-action(20k), escape-rich": band(b20 - f20),
         "balanced(20k) - balanced(13,592), escape-rich": band(b20 - b13)}
passed = True
for name, (mean, lo, hi) in tests.items():
    ok = lo > 0
    passed &= ok
    print(f"  {name:<48}{mean:+.4f} [{lo:+.4f}, {hi:+.4f}]  {'PASS' if ok else 'FAIL'}")
print()
print("VERDICT: replicate balanced and full-action once more" if passed else
      "VERDICT: stop all terminal-data arms. Do not extend to 27,184 -- the full-action "
      "arm already saw every (root, action) pair about 25 times, so insufficient "
      "repetition is not a credible explanation.")
EOF
  ;;
queue)
  # everything that runs once both lanes are done, in dependency order
  while pgrep -f "[r]un_seed2_matrix.sh [ab]" > /dev/null; do sleep 60; done
  echo "both lanes finished; scoring"; echo
  bash "$0" score
  echo; echo "########## preregistered continuation gate ##########"; echo
  bash "$0" gate
  echo; echo "########## root x action decomposition ##########"; echo
  for arm in "production_1b_s2 production_s2" "terminal_factual_s2 factual_s2" \
             "terminal_counterfactual_s2 counterfactual_s2" \
             "terminal_full17_s2 full17_s2" "terminal_balanced_s2 balanced_s2"; do
    set -- $arm
    [ -f "$1/world.pt" ] || { echo "-- $2 unfinished, skipped"; continue; }
    [ -f "action_interaction_$2.json" ] && { echo "-- $2 already decomposed"; continue; }
    echo "### $2 ###"; "$PY" probe_action_interaction.py --folder "$1" --tag "$2" || exit 1
  done
  echo; echo "########## regression gates, per-root ##########"; echo
  for arm in "production_1b_s2 production_s2" "terminal_factual_s2 factual_s2" \
             "terminal_counterfactual_s2 counterfactual_s2" \
             "terminal_full17_s2 full17_s2" "terminal_balanced_s2 balanced_s2"; do
    set -- $arm
    [ -f "$1/world.pt" ] || { echo "-- $2 unfinished, skipped"; continue; }
    [ -f "production_1b_evaluation_$2.json" ] && { echo "-- $2 already gated"; continue; }
    echo "### $2 ###"; "$PY" evaluate_production_1b.py --world "$1/world.pt" --tag "$2" \
        2>&1 | tail -12
  done
  echo; echo "queue complete"
  ;;
c)
  # the two data-composition arms. Everything except root sampling matches the existing
  # full-17 terminal arm, which is the terminal-only control.
  arm broad_uniform_s2   --arm counterfactual --roots broad \
                         --terminal-roots 4 --terminal-actions 17 || exit 1
  ;;
d)
  arm broad_regime_s2    --arm counterfactual --roots broad --regime-balance \
                         --terminal-roots 4 --terminal-actions 17 || exit 1
  ;;
e)
  # broad-uniform only, at the DEFAULT world seed, against the existing production_1b
  # control trained at that same seed. Regime balancing is not replicated: it did not
  # move the primary death endpoint despite 4.4x the escape-root repetition.
  if finished broad_uniform_s0/training_report.json steps; then
    echo "== broad_uniform_s0 already at $STEPS"
  else
    echo "== broad_uniform_s0"
    "$PY" train_terminal_arms.py --steps "$STEPS" --out broad_uniform_s0 \
        --arm counterfactual --roots broad --terminal-roots 4 --terminal-actions 17 || exit 1
  fi
  ;;
*) echo "lane must be a, b, c, d, e, score, gate or queue"; exit 2;;
esac
echo "lane $1 complete"
