#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export TRITON_F32_DEFAULT=ieee
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOSS_RUN=artifacts/eda/oracle_v1/loss_intervention
LOSS_LOG=artifacts/eda/levers_logs/loss_intervention_eval.log
{
    date '+START %Y-%m-%d %H:%M:%S %Z'
    for update in 0 500 1500 3000; do
        printf -v suffix '%04d' "$update"
        while [[ ! -f "$LOSS_RUN/no_max_u$suffix.pt" ]]; do
            if ! systemctl --user is-active --quiet oracle_loss_intervention_20261012; then
                printf 'FAILED: training stopped before milestone %s\n' "$suffix"
                exit 1
            fi
            sleep 10
        done
        if [[ "$update" == 0 ]]; then
            # Identical initial model/conditioning weights: infer baseline once.
            .venv/bin/python artifacts/experiments/20261012_oracle/loss_intervention_eval.py \
                --updates "$update" --prefix loss_intervention --arms D
        else
            .venv/bin/python artifacts/experiments/20261012_oracle/loss_intervention_eval.py \
                --updates "$update" --prefix loss_intervention
        fi
    done
    date '+SUCCESS %Y-%m-%d %H:%M:%S %Z'
} >> "$LOSS_LOG" 2>&1
