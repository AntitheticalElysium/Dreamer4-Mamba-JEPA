"""Post-hoc CPU source/destination dependence diagnostic; never a repair test.

Independent per-cell permutations preserve all five empirical cell marginals.
Changes in joint occupancy measure observed dependence, not latent causal factors.
Frozen experiment scripts, readers, thresholds and criteria remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
import torch

import direct_analysis as A
import direct_coherence as C

OUT = A.OUT / "diagnostics/joint_shuffle"


def stats(flags):
    """Population moments of the empirical five-cell source neighborhood."""
    x = flags.double()
    probability = x.mean(0)
    count = x.sum(-1)
    count_variance = float(count.var(unbiased=False))
    independent_variance = float((probability * (1 - probability)).sum())
    valid_move = (count == 1) & (~flags[:, 0])
    return {
        "samples": len(x), "cell_probability": probability.tolist(),
        "count_mean": float(count.mean()), "count_variance": count_variance,
        "sum_marginal_bernoulli_variance": independent_variance,
        "conservation_fraction": 1 - count_variance / independent_variance if independent_variance > 0 else None,
        "absent_probability": float((count == 0).double().mean()),
        "multiple_probability": float((count > 1).double().mean()),
        "duplicate_source_and_destination_probability": float((flags[:, 0] & flags[:, 1:].any(1)).double().mean()),
        "valid_stay_probability": float(((count == 1) & flags[:, 0]).double().mean()),
        "valid_move_probability": float(valid_move.double().mean()),
        "valid_outcome_probability": float((count == 1).double().mean()),
    }


def permuted(flags, generator):
    return torch.stack([flags[torch.randperm(len(flags), generator=generator), c]
                        for c in range(flags.shape[-1])], 1)


def means(rows):
    keys = [k for k, v in rows[0].items() if not isinstance(v, list)]
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
            if any(r[k] is not None for r in rows) else None for k in keys}


def selftest():
    flags = torch.nn.functional.one_hot(torch.arange(256) % 4, 5).bool()
    original = stats(flags)
    shuffled = permuted(flags, torch.Generator().manual_seed(99))
    assert torch.equal(flags.sum(0), shuffled.sum(0))
    check = stats(shuffled)
    assert original["conservation_fraction"] == 1 and original["valid_outcome_probability"] == 1
    assert check["absent_probability"] > 0 and check["multiple_probability"] > 0
    assert check["conservation_fraction"] < .2
    return {"marginals_preserved_exactly": True, "coherent": original, "shuffled": check}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026101403)
    args = parser.parse_args()
    torch.set_num_threads(4)
    directory = A.OUT / "eval_exploratory_u3000"
    analysis = json.loads((directory / "analysis.json").read_text())
    units = [u for u in analysis["arms"]["energy"]["source_units"] if u["primary"] and u["reader_clean"]]
    bundle = A.exploratory_bundle()
    files = [Path(__file__), Path(A.__file__), Path(C.__file__), directory / "analysis.json",
             directory / "pred_point.pt", directory / "pred_energy.pt", A.BANK, A.REAL, A.METADATA,
             A.BASE / "panel.pt", A.BASE / "learned_innovation/real_futures/fan.pt"]
    contract = {"format": "direct_joint_shuffle_v1", "scope": "post-hoc exploratory dependence diagnostic",
                "unit": "14 preselected isolated full-support source neighborhoods; no generated sample selection",
                "intervention": "Independently permute each cell across samples, preserving empirical cell marginals exactly",
                "repetitions": args.repetitions, "seed": args.seed, "source_units": units,
                "sources_inputs": {str(p.relative_to(A.ROOT)): A.sha(p) for p in files},
                "limits": ["Reader-level occupancy, not a proof of physical latent identity or generative causal independence",
                           "32 real branches and 64 generated branches have different finite-fan resolution",
                           "Hidden-phase zombie and skeleton conditional probabilities may differ from H15 belief",
                           "No new model training, no threshold fitting, no replacement for declared verdict"]}
    cid = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "contract.json").exists():
        assert json.loads((OUT / "contract.json").read_text()) == contract
    else:
        A.atomic(contract, OUT / "contract.json")
    validation = selftest()
    bank, head = A.load(A.BANK), A.load(A.REAL)["heads"]["real"]
    decoded_real = C.decode(bundle["fan_tokens"], bank, head)["mob"]
    conditions = {"real_truth": bundle["fan_layout"].bool(),
                  "real_reader": decoded_real["real"], "real_bank": decoded_real["bank"]}
    for arm in ("point", "energy"):
        decoded = C.decode(A.load(directory / f"pred_{arm}.pt")["predictions"], bank, head)["mob"]
        conditions.update({arm + "_" + reader: flags for reader, flags in decoded.items()})
    result = {"contract_id": cid, "contract": contract, "validation": validation, "conditions": {}}
    for condition, flags in conditions.items():
        path = OUT / f"{condition}.json"
        if path.exists():
            report = json.loads(path.read_text())
            assert report["contract_id"] == cid and report["condition"] == condition
        else:
            per_source = []
            for j, unit in enumerate(units):
                i, points, kind = unit["root"], unit["points"], unit["kind"]
                local = flags[i, :, points, kind]
                actual = stats(local)
                draws = []
                generator = torch.Generator().manual_seed(args.seed + j)
                for _ in range(args.repetitions):
                    shuffled = permuted(local, generator)
                    assert torch.equal(local.sum(0), shuffled.sum(0))
                    draws.append(stats(shuffled))
                per_source.append({"root": i, "kind": C.LAYOUT_TYPES[kind], "source": points[0],
                                   "actual": actual, "shuffled_mean": means(draws),
                                   "shuffled_draws": draws})
            groups = {"all": per_source, "cow": [r for r in per_source if r["kind"] == "cow"],
                      "zombie": [r for r in per_source if r["kind"] == "zombie"],
                      "skeleton": [r for r in per_source if r["kind"] == "skeleton"]}
            report = {"contract_id": cid, "condition": condition, "per_source": per_source,
                      "groups": {g: {"sources": len(rows), "actual": means([r["actual"] for r in rows]),
                                     "shuffled": means([r["shuffled_mean"] for r in rows])}
                                 for g, rows in groups.items()}}
            A.atomic(report, path)
        result["conditions"][condition] = report
        print(json.dumps({"stage": "condition_done", "condition": condition, "groups": report["groups"]}), flush=True)
    A.atomic(result, OUT / "analysis.json")
    print(json.dumps({"stage": "done", "contract_id": cid, "path": str(OUT / "analysis.json")}), flush=True)


if __name__ == "__main__":
    main()
