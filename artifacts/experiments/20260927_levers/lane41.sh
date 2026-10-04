source artifacts/experiments/20260927_levers/lib.sh
# Diagnostics of the unexplained (2026-10-02 ~18:00, user's review; no training). Readings declared here before any ran:
# D1 (item 1) all-token precision across snapshots: check_allprobe on every 6k snapshot of both seeds (consfit alongside).
#   representation_first = h's linear AP at least doubles (vs the 6k snapshot) at a snapshot BEFORE the one where held caught first
#   exceeds 0.1 (precision leads catching); simultaneous = both in the same 6k interval; catch_first = catching rises first.
# D3 (items 3, 7) the same probe on the dosed backbones (E14c s7 / s8, E14d s7, linear s7): end_to_end_precise = h linear AP >= 0.6;
#   refit_paradox_holds = AP >= 0.6 on E14c s7 while its uniform-head refit caught 0.000 (precision is then not sufficient under
#   the uniform loss).
# D4/D6 (items 4, 6) check_e14c_cost on E14d s7 and linear s7; check_e14c_grad on E14d s7 (dominance at x31).
# D2 (item 2) subst16 oracle arms (cons, enter, cons+enter, realign) on the 36k s7 world, against the 18k world's (lane28/28b):
#   consequence_gain_shrinks = the cons arm's relative reduction of ever_position_wrong on 36k is at most half the 18k world's
#   (-37%): the 36k world already carries most of what the consequence oracle gave.
W=artifacts/eda/levers_tworlds_v1
S7="$W/corrt_raw_teacher_s7_u36000_at6000.pt $W/corrt_raw_teacher_s7_u36000_at12000.pt $W/corrt_raw_teacher_s7_u36000_at18000.pt $W/corrt_raw_teacher_s7_u36000_at24000.pt $W/corrt_raw_teacher_s7_u36000_at30000.pt $W/corrt_raw_teacher_s7_u36000.pt"
until [ -f $W/corrt_raw_teacher_s8_u36000.pt ]; do sleep 60; done
S8="$W/corrt_raw_teacher_s8_u36000_at6000.pt $W/corrt_raw_teacher_s8_u36000_at12000.pt $W/corrt_raw_teacher_s8_u36000_at18000.pt $W/corrt_raw_teacher_s8_u36000_at24000.pt $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_u36000.pt"
job allprobe_snapshots 1500 $PY $L/check_allprobe.py $S7 $S8
job consfit_snapshots 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u36000_at6000.pt $W/corrt_raw_teacher_s7_u36000_at12000.pt $W/corrt_raw_teacher_s8_u36000_at6000.pt $W/corrt_raw_teacher_s8_u36000_at12000.pt $W/corrt_raw_teacher_s8_u36000_at18000.pt $W/corrt_raw_teacher_s8_u36000_at24000.pt $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_u36000.pt
job allprobe_dosed 1500 $PY $L/check_allprobe.py $W/corrt_raw_teacher_s7_mask1_skip_u18000.pt $W/corrt_raw_teacher_s8_mask1_skip_u18000.pt $W/corrt_raw_teacher_s7_mask0.1_skip_u18000.pt $W/corrt_raw_teacher_s7_mask1_u18000.pt
job e14d_cost 1200 $PY $L/check_e14c_cost.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s7_mask0.1_skip_u18000.pt $W/corrt_raw_teacher_s7_mask1_u18000.pt
job e14d_grad 1800 $PY $L/check_e14c_grad.py $W/corrt_raw_teacher_s7_mask0.1_skip_u18000.pt
[ -f $L/evals/subst16_corrt_raw_teacher_s7_u36000_oracle.json ] || job subst16_oracle_36k_s7 2000 $PY $L/subst16.py $W/corrt_raw_teacher_s7_u36000.pt --arms base,cons,enter,cons+enter,realign --tag oracle
echo "$(date '+%F %T') LANE41_DONE" >> $LOGDIR/lanes.log
