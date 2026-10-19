source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 18:4x: SECOND-SEED replication (standing >= 2 init seeds rule) of the seed-7-only training-dynamics findings, L1 side
# (no D-loss seed-8 run exists): the M6 seed-8 snapshots 6k..36k (P-D's L1 curves: HUD-change stall, depth-16 mob erasure growing
# with training) and the seed-8 L16 world (statoracle's depth-1 census match, proper scores). Measured need: snapshot export 2.1 GB.
V=artifacts/eda/levers_tworlds_v1
job sb_export_s8 2300 $PY $L/rollout_export.py --world M6s8_6k=$V/corrt_raw_teacher_s8_fmamba_u36000_at6000.pt --world M6s8_12k=$V/corrt_raw_teacher_s8_fmamba_u36000_at12000.pt --world M6s8_18k=$V/corrt_raw_teacher_s8_fmamba_u36000_at18000.pt --world M6s8_24k=$V/corrt_raw_teacher_s8_fmamba_u36000_at24000.pt --world M6s8_30k=$V/corrt_raw_teacher_s8_fmamba_u36000_at30000.pt --world M6s8_36k=$V/corrt_raw_teacher_s8_fmamba_u36000.pt --world M16s8=$V/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE181_DONE" >> $LOGDIR/lanes.log
