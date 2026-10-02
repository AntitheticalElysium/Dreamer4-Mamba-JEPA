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
# Order (one 3.1 GB training at a time): skip s7, skip s8, E14b's 36k s7 (declared in lane32.sh, moved here 2026-10-02 11:15),
# linear s7, linear s8, 36k s8
train() { N=$1; shift; [ -f $W/$N.pt ] || job tworld_$N 3100 $PY $L/tworld.py --head corrt --loss teacher "$@"; }
train corrt_raw_teacher_s7_mask1_skip_u18000 --seed 7 --updates 18000 --weight mask1 --skip
train corrt_raw_teacher_s8_mask1_skip_u18000 --seed 8 --updates 18000 --weight mask1 --skip
train corrt_raw_teacher_s7_u36000 --seed 7 --updates 36000 --snapshots
train corrt_raw_teacher_s7_mask1_u18000 --seed 7 --updates 18000 --weight mask1
train corrt_raw_teacher_s8_mask1_u18000 --seed 8 --updates 18000 --weight mask1
train corrt_raw_teacher_s8_u36000 --seed 8 --updates 36000 --snapshots
job consfit_36k 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u36000_at18000.pt $W/corrt_raw_teacher_s7_u36000_at24000.pt \
  $W/corrt_raw_teacher_s7_u36000_at30000.pt $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u36000_at18000.pt \
  $W/corrt_raw_teacher_s8_u36000_at24000.pt $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_u36000.pt
echo "$(date '+%F %T') LANE35_DONE" >> $LOGDIR/lanes.log
