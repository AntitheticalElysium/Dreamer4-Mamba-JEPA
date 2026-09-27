# Shared by the lane scripts. Every job: waits for GPU admission (free memory >= its declared need + 256 MiB,
# checked twice 20 s apart), then runs; a CUDA-OOM failure is retried (6 x 3 min), any other failure stops the lane.
# Jobs are idempotent (callers skip finished outputs), so a lane can be relaunched after any interruption.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
PY=.venv/bin/python
L=artifacts/experiments/20260927_levers
LOGDIR=artifacts/eda/levers_logs
export TRITON_F32_DEFAULT=ieee PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
admit() { # need_mib
  while true; do
    if [ "$(free_mib)" -ge $(( $1 + 256 )) ]; then sleep 20; [ "$(free_mib)" -ge $(( $1 + 256 )) ] && return 0; fi
    sleep 30
  done
}
job() { # name need_mib cmd...
  # Admission is serialized by a lock held until this job's own GPU allocation is visible (its pid appears in
  # nvidia-smi and 30 s pass), so two jobs can never pass admission against the same free memory while they are
  # still loading data. need_mib 0 = CPU-only (no lock, no admission).
  local name=$1 need=$2; shift 2
  local log=$LOGDIR/$name.log
  for attempt in 1 2 3 4 5 6; do
    if [ "$need" -gt 0 ]; then
      exec 9>>$LOGDIR/gpu.lock; flock -x 9
      admit $need
    fi
    echo "$(date '+%F %T') START $name attempt $attempt" >> $LOGDIR/lanes.log
    "$@" >> "$log" 2>&1 &
    local pid=$!
    if [ "$need" -gt 0 ]; then
      for i in $(seq 1 60); do
        kill -0 $pid 2>/dev/null || break
        nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qx "$pid" && { sleep 30; break; }
        sleep 5
      done
      flock -u 9; exec 9>&-
    fi
    wait $pid; local code=$?
    if [ $code -eq 0 ]; then echo "$(date '+%F %T') DONE $name" >> $LOGDIR/lanes.log; return 0; fi
    if tail -40 "$log" | grep -q "out of memory"; then
      echo "$(date '+%F %T') OOM $name attempt $attempt" >> $LOGDIR/lanes.log; sleep 120; continue
    fi
    echo "$(date '+%F %T') FAILED $name (exit $code)" >> $LOGDIR/lanes.log; exit 1
  done
  echo "$(date '+%F %T') FAILED $name after OOM retries" >> $LOGDIR/lanes.log; exit 1
}
