#!/usr/bin/env bash
# Waits for the three T2 lanes to finish, smoke-tests every T1 structure (100 steps, batch 16), and launches the three T1
# lanes only if all smoke tests pass. Writes t1_smoke_report.txt.
set -u
R=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA; X=$R/artifacts/experiments/20261023_system_design
OUT=$R/artifacts/eda/system_design_v1; PY=$R/.venv/bin/python
while systemctl --user is-active -q t2-lane0 || systemctl --user is-active -q t2-lane1 || systemctl --user is-active -q t2-lane2; do sleep 30; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ok=1; : > $OUT/t1_smoke_report.txt
for arch in G S A O E Estar; do
  if $PY $X/t1_train.py --data $OUT/t1_data --out $OUT/t1_smoke/$arch --arch $arch --seed 1 --steps 100 --batch 16 >> $OUT/t1_smoke_report.txt 2>&1; then
    echo "$arch OK $(tail -1 $OUT/t1_smoke/$arch/train_log.jsonl)" >> $OUT/t1_smoke_report.txt
  else echo "$arch FAILED" >> $OUT/t1_smoke_report.txt; ok=0; fi
done
[ $ok = 1 ] || { echo "smoke failed; T1 not launched" >> $OUT/t1_smoke_report.txt; exit 1; }
for i in 0 1 2; do systemd-run --user --unit t1-lane$i -q -p MemoryMax=8G --working-directory=$R -E OMP_NUM_THREADS=3 $X/t1_lane.sh $X/t1_lane$i.txt; done
echo "T1 lanes launched" >> $OUT/t1_smoke_report.txt
