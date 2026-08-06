#!/usr/bin/env bash
# Deployment-noise check, deferred until queue10 releases the GPU.
# Re-evaluates the queue9 slow checkpoints under two DIFFERENT policy-sampling
# seed bases. Their bootstrap CIs capture environment-seed uncertainty only, so
# a single sampling schedule cannot show whether the slow-T gain survives
# stochastic deployment.
#
# --allow-implementation-drift is used deliberately and is justified: the only
# change since those checkpoints is the construction ORDER of the mamba2
# substitution. state_dict keys and shapes are identical (243 tensors), so a
# strict load fully determines every parameter and the forward path is
# unchanged. Verified before enabling the flag, not assumed.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue10
PY=.venv/bin/python
while pgrep -f "craftax_queue10.sh" > /dev/null; do sleep 60; done
for base in 7100000 7200000; do
  for arm in t_jepa m_jepa; do
    n="resample_${arm}_${base}"
    echo "=== [$(date -u +%H:%M:%S)] START $n ===" | tee -a "$LOGDIR/queue10.log"
    if $PY -u reviews/artifacts/craftax_achievement_run.py \
         --run-dir outputs/d4_mamba_jepa/craftax_slowenc_v1 --arm "$arm" \
         --policy-seed-base "$base" --allow-implementation-drift \
         --output reviews/artifacts/craftax_executed_lr/${n}.json \
         > "$LOGDIR/${n}.log" 2>&1; then
      echo "=== [$(date -u +%H:%M:%S)] OK    $n ===" | tee -a "$LOGDIR/queue10.log"
    else
      echo "=== [$(date -u +%H:%M:%S)] FAIL  $n ===" | tee -a "$LOGDIR/queue10.log"
    fi
  done
done
echo "=== [$(date -u +%H:%M:%S)] QUEUE11 COMPLETE ===" | tee -a "$LOGDIR/queue10.log"
