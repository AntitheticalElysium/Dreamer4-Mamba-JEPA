"""Resume Flow probes from a completed, contract-pinned feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.localize_counterfactual import binary_metrics
from artifacts.localize_counterfactual_interaction import report_score
from artifacts.localize_direct_transition_stages import _resumable_linear_probe
from artifacts.localize_flow_phase1b import FlowForkData, _latent_error
from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from d4mj.config import Config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--linear-steps", type=int, default=600)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()

    payload = torch.load(args.features, weights_only=False)
    extraction = payload["contract"]
    data = FlowForkData(**payload["data"])
    replay = payload["replay"]
    config = Config(transition="flow", time_mixer="attention")
    signal_levels = tuple(extraction["signal_levels"])
    if config.tau_ctx_index not in signal_levels:
        raise ValueError("extraction omitted the production signal level")
    expected = replay["terminal_opportunity_states"] * config.n_actions
    if len(data.target) != expected:
        raise ValueError("feature row count does not match the extraction contract")

    contract = {
        "version": "flow-phase1b-feature-report-v1",
        "features": file_digest(args.features),
        "extraction_contract": extraction,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/localize_direct_transition_stages.py"),
            Path("artifacts/localize_counterfactual.py"),
            Path("artifacts/localize_counterfactual_interaction.py"),
        ),
        "probe": "action-centered leave-one-pre-action-state-out linear",
        "seeds": args.seeds,
        "linear_steps": args.linear_steps,
        "permutations": args.permutations,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probes").mkdir(exist_ok=True)
    contract_path = args.out / "report_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Flow feature-report contract changed")
    else:
        atomic_json(contract_path, contract)

    features = {
        "observed_clean_latent": data.observed_latent,
        "generated_latent_first": data.generated_latent_first,
        "generated_latent_mean": data.generated_latent_mean,
        "generated_readout_first": data.generated_readout_first,
        "generated_readout_mean": data.generated_readout_mean,
        "observed_readout_first": data.conditioned_readout_first[config.tau_ctx_index],
        "observed_readout_mean": data.conditioned_readout_mean[config.tau_ctx_index],
    }
    for level in signal_levels:
        if level != config.tau_ctx_index:
            features[f"conditioned_readout_mean_tau{level}"] = data.conditioned_readout_mean[level]

    probe_seeds = [config.seed + 4000 + index for index in range(args.seeds)]
    probes = {}
    for index, (name, feature) in enumerate(features.items()):
        probe_contract = {
            "version": contract["version"],
            "stage": "equalized_flow",
            "feature": name,
            "features": contract["features"],
            "seeds": probe_seeds,
            "steps": args.linear_steps,
            "lr": 3e-3,
            "weight_decay": 1e-3,
        }
        probability = _resumable_linear_probe(
            feature,
            data.target,
            data.action,
            data.group,
            config,
            seeds=probe_seeds,
            steps=args.linear_steps,
            checkpoint=args.out / "probes" / f"{name}.pt",
            contract=probe_contract,
        )
        probes[name] = {
            "binary": binary_metrics(probability, data.target),
            "same_action": report_score(
                probability,
                data.target,
                data.action,
                permutations=args.permutations,
                seed=config.seed + 7000 + index,
            ),
        }

    def score(probability, seed):
        return {
            "binary": binary_metrics(probability, data.target),
            "same_action": report_score(
                probability,
                data.target,
                data.action,
                permutations=args.permutations,
                seed=seed,
            ),
        }

    production = {
        "generated_first": score(data.generated_death_first, config.seed + 8000),
        "generated_mean": score(data.generated_death_mean, config.seed + 8001),
        "conditioned_first": {},
        "conditioned_mean": {},
    }
    for level in signal_levels:
        production["conditioned_first"][str(level)] = score(
            data.conditioned_death_first[level], config.seed + 8100 + level
        )
        production["conditioned_mean"][str(level)] = score(
            data.conditioned_death_mean[level], config.seed + 8200 + level
        )

    report = {
        "contract": contract,
        "replay": replay,
        "latent_prediction_error": {
            "first_sample": _latent_error(
                data.generated_latent_first, data.observed_latent, data.target
            ),
            "sample_mean": _latent_error(
                data.generated_latent_mean, data.observed_latent, data.target
            ),
        },
        "sample_variance": {
            "generated_latent": float(data.generated_latent_variance.mean()),
            "generated_readout": float(data.generated_readout_variance.mean()),
            "conditioned_readout": {
                str(level): float(data.conditioned_readout_variance[level].mean())
                for level in signal_levels
            },
        },
        "production_head": production,
        "fresh_probes": probes,
    }
    atomic_json(args.out / "report.json", report)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()

