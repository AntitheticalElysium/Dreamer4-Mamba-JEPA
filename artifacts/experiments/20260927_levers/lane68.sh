source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: cross-scroll recall learning curves (check_recall on saved snapshots). Readings, declared before running (computed
# from check_recall.log with exactly these rules):
#   mamba_recall_edge_s8   s8: fmamba's recall_capture - attention's >= 0.10 at 30k (the latest snapshot both s8 runs have)
#   mamba_recall_edge_both at every matched snapshot from 18k on, fmamba's capture > attention's at both seeds
#   attention_catches_up   attention s7 at 100k (and s8 at 100k) reaches the s7 fmamba 36k capture (0.778)
W=artifacts/eda/levers_tworlds_v1
args=""
for s in 7 8; do
  for u in 6000 12000 18000 24000 30000; do args="$args $W/corrt_raw_teacher_s${s}_u36000_at${u}.pt $W/corrt_raw_teacher_s${s}_fmamba_u36000_at${u}.pt"; done
  args="$args $W/corrt_raw_teacher_s${s}_u36000.pt $W/corrt_raw_teacher_s${s}_u50000.pt $W/corrt_raw_teacher_s${s}_u100000.pt"
done
job check_recall_curve 2000 $PY $L/check_recall.py $args $W/corrt_raw_teacher_s7_fmamba_u36000.pt
echo "$(date '+%F %T') LANE68_DONE" >> $LOGDIR/lanes.log
