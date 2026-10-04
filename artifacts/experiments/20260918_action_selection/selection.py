"""Does a refitted world model choose the safer action, where global AUC says it adds nothing?

The generated-fit rescore compared conditions by macro AUC and found every arm sitting on its
root+action control.  AUC is a global ranking over all (root, action) pairs pooled together;
safe-action choice is a ranking *within* a root, over its 17 forks.  A model can be no better
at telling risky states from safe ones in general, and still be better at telling which fork of
*this* state kills you.  Those are different questions and the second is the one a planner asks.

So this scores `_choice_summary` -- the gate's own safe-fork selection -- under each readout
condition, on the same fitted probes the rescore used, and adds a paired cluster bootstrap on
the difference so "beats its control" is a measured interval rather than two point estimates
compared by eye.

Read-only over published M03 features.  Changes no gate output and authorizes nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.data import _sha256, atomic_manifest
from d4mj.m03.cache import ArtifactCache, resolve_payload, use_cache
from d4mj.m03.gate import (OUTCOME_BINARY, M03Settings, _choice_summary, _fit_probe_many,
                           _one_hot_actions)

MINIMIZE = {"death": True, "damage": True, "reward_positive": False}


def paired_difference(scores_a, scores_b, truth, roots, index, *, minimize, settings, seed):
    """Cluster bootstrap on (A - B) selection rate, resampling episodes once per draw.

    Both conditions are evaluated on the *same* resampled episodes, which is what makes the
    interval a paired one; drawing separately would widen it with variance the comparison
    does not contain.
    """
    labels = truth[..., index].bool()
    eligible = labels.any(1) & (~labels).any(1)
    desired = ~labels if minimize else labels

    def rate(scores, rows):
        valid = eligible[rows]
        if not bool(valid.any()):
            return None
        chosen = scores[rows][..., index].argmin(1) if minimize else scores[rows][..., index].argmax(1)
        return float(desired[rows, chosen][valid].float().mean())

    everything = torch.arange(len(roots))
    point_a, point_b = rate(scores_a, everything), rate(scores_b, everything)
    if point_a is None or point_b is None:
        return None
    keys = roots.unique(sorted=True)
    groups = [torch.where(roots == key)[0] for key in keys]
    rng = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(settings.bootstrap_draws):
        sampled = torch.randint(len(groups), (len(groups),), generator=rng)
        rows = torch.cat([groups[number] for number in sampled])
        a, b = rate(scores_a, rows), rate(scores_b, rows)
        if a is not None and b is not None:
            samples.append(a - b)
    if len(samples) < .95 * settings.bootstrap_draws:
        return {"difference": point_a - point_b, "interval": None, "a": point_a, "b": point_b}
    bounds = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"a": point_a, "b": point_b, "difference": point_a - point_b, "interval": bounds,
            "excludes_zero": bool(bounds[0] > 0 or bounds[1] < 0)}


def arm_selection(features, split_rows, settings, device, seed_offset):
    train, dev = split_rows["train"], split_rows["dev"]
    train_truth = train["outcomes"].reshape(-1, len(OUTCOME_BINARY)).float().to(device)
    dev_truth = dev["outcomes"]
    roots = dev["episode"]
    at = _one_hot_actions(len(train["outcomes"]), device=device)
    ad = _one_hot_actions(len(dev["outcomes"]), device=device)

    def pack(split, field, action):
        return torch.cat((features[split][field].flatten(0, 1).float().to(device), action), 1)

    root_train = torch.cat((features["train"]["projected"].float().to(device).repeat_interleave(17, 0), at), 1)
    root_dev = torch.cat((features["dev"]["projected"].float().to(device).repeat_interleave(17, 0), ad), 1)
    out = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        fitted = {
            "observed_fit_on_generated": _fit_probe_many(
                pack("train", "observed_successor", at), train_truth,
                {"d": pack("dev", "generated_successor", ad)}, settings, hidden=hidden, binary=True)["d"],
            "generated_fit_on_generated": _fit_probe_many(
                pack("train", "generated_successor", at), train_truth,
                {"d": pack("dev", "generated_successor", ad)}, settings, hidden=hidden, binary=True)["d"],
            "root_action": _fit_probe_many(root_train, train_truth, {"d": root_dev},
                                           settings, hidden=hidden, binary=True)["d"],
            "action_only": _fit_probe_many(at, train_truth, {"d": ad},
                                           settings, hidden=hidden, binary=True)["d"],
        }
        shaped = {k: v.reshape(-1, 17, len(OUTCOME_BINARY)).cpu() for k, v in fitted.items()}
        family_out = {"conditions": {}, "paired_vs_root_action": {}}
        for name in ("death", "reward_positive"):
            index = OUTCOME_BINARY.index(name)
            family_out["conditions"][name] = {
                k: _choice_summary(v, dev_truth, roots, index, minimize=MINIMIZE[name], settings=settings)
                for k, v in shaped.items()}
            family_out["paired_vs_root_action"][name] = {
                k: paired_difference(shaped[k], shaped["root_action"], dev_truth, roots, index,
                                     minimize=MINIMIZE[name], settings=settings,
                                     seed=settings.seed + seed_offset + index)
                for k in ("generated_fit_on_generated", "observed_fit_on_generated")}
        out[family] = family_out
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.label}.json"
    if destination.exists():
        raise FileExistsError(f"selection: refusing to replace {destination}")

    settings = M03Settings()
    sidecar = torch.load(args.run / "sidecar/sidecar.probe_only.pt", map_location="cpu", weights_only=False)
    cache = ArtifactCache(args.cache, settings=settings, device=args.device)
    report = {"schema": "d4mj_action_selection_v1", "run": str(args.run.resolve()),
              "question": "within-root safe-fork choice under each readout condition, with a paired "
                          "cluster bootstrap against the root+action control",
              "sidecar_sha256": _sha256(args.run / "sidecar/sidecar.probe_only.pt"),
              "script_sha256": _sha256(Path(__file__)), "arms": {}, "m4_authorized": False}
    with use_cache(cache):
        for offset, arm in enumerate(("raw", "tc", "direct_mamba", "direct_attention")):
            paths = {s: args.run / f"features/{arm}.{s}.pt" for s in ("train", "dev")}
            if not all(p.exists() for p in paths.values()):
                continue
            features = {s: resolve_payload(torch.load(p, map_location="cpu", weights_only=False))["features"]
                        for s, p in paths.items()}
            if "generated_successor" not in features["train"]:
                continue
            report["arms"][arm] = arm_selection(features, sidecar["splits"], settings, args.device, offset * 37)
            print(json.dumps({"stage": "arm_complete", "arm": arm}), flush=True)
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
