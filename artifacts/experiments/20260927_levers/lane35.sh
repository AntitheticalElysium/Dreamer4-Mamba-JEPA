source artifacts/experiments/20260927_levers/lib.sh
# E14c (2026-10-02): end-to-end training with the head-only fix, the Craftax-labelled UPPER BOUND (diagnostic): does catching the
# consequences improve imagination and decisions? corrt, Raw, teacher loss, 18,000 updates, seeds 7 and 8, the recipe of
# corrt_raw_teacher_s{7,8}_u18000 except: --weight mask1 (CGSReg-form term on the faced tile of every DO / place attempt,
# lambda 1, dose x304) with --skip (head reads the raw local neighbourhood) [primary], or with the linear head [is --skip needed].
# Head-only evidence (E14a, both seeds): skip_mask1 caught 0.63 / 0.65 at +7% / +9% all-token L1; linear mask1 0.54 at +21% (s7).
# Readings, declared before any of these worlds trained (two-seed rule; comparator = the same-seed teacher u18000 world):
#   c_learned     consfit held strict caught >= 0.5 at both seeds
#   c_cost        teval onestep_all ratio (arm / teacher) <= 1.10 at both seeds
#   c_position    subst16 base ever_position_wrong at least 20% lower (relative) at both seeds (E13's oracle consequence
#                 substitution: -37% at both seeds)
#   c_depth16     compare.py gen_16 (arm - teacher) not resolved > 0 at both seeds
#   c_decision    dpanel gen1 or gen2 (arm - teacher) resolved > 0 at both seeds
#   skip_needed   the skip arm passes c_learned and c_cost and the linear arm fails one of them
# Monotone (revealed terrain) is reported for both.
W=artifacts/eda/levers_tworlds_v1
# Order: two lanes in parallel (2026-10-02 12:45: measured peak 1.9 GB reserved for teacher + skip + mask1, 1.7 GB plain;
# admission 2,400 MiB): this lane = seed 7 (skip, 36k, linear); lane35b.sh = seed 8 (skip, linear, 36k) + consfit_36k
train() { N=$1; shift; [ -f $W/$N.pt ] || job tworld_$N 2400 $PY $L/tworld.py --head corrt --loss teacher "$@"; }
train corrt_raw_teacher_s7_mask1_skip_u18000 --seed 7 --updates 18000 --weight mask1 --skip
train corrt_raw_teacher_s7_u36000 --seed 7 --updates 36000 --snapshots
# 2026-10-02 15:45: the E14c evaluations of both skip worlds (lane36) first
until [ -f $L/evals/monotone_corrt_raw_teacher_s8_mask1_skip_u18000.json ]; do sleep 60; done
train corrt_raw_teacher_s7_mask1_u18000 --seed 7 --updates 18000 --weight mask1
echo "$(date '+%F %T') LANE35_DONE" >> $LOGDIR/lanes.log
