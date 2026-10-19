# 2026-10-10 10:07: 10-10 PROGRAM Part II-A (encoder), CPU, one job at a time (systemd unit, MemoryMax 10G): the full Raw run (five futures,
# noise floor), then the encoder comparison on identical frames (root + factual future): raw, ldad1, ldad10, tc.
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export CUDA_VISIBLE_DEVICES=
for E in full raw ldad1 ldad10 tc; do
  if [ $E = full ]; then A=""; else A="--encoder $E"; fi
  echo "$(date '+%F %T') START encdiag_$E" >> artifacts/eda/levers_logs/lanes.log
  .venv/bin/python artifacts/experiments/20260927_levers/encoder_diag.py $A > artifacts/eda/levers_logs/encdiag_$E.log 2>&1 \
    && echo "$(date '+%F %T') DONE encdiag_$E" >> artifacts/eda/levers_logs/lanes.log \
    || echo "$(date '+%F %T') FAILED encdiag_$E" >> artifacts/eda/levers_logs/lanes.log
done
echo "$(date '+%F %T') CPU172_DONE" >> artifacts/eda/levers_logs/lanes.log
