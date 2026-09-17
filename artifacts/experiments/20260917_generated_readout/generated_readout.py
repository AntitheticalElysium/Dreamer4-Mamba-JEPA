"""Is the information missing from generated latents, or is the decoder failing on them?

M03 fits one decoder on `train_observed_successor` and applies it to observed, generated
and reset latents alike.  That is deliberate -- `_fit_probe_many` says so -- and it is the
right design for asking whether generated latents land in the *same* decodable space.  It
cannot separate "the information is absent" from "the information is there, in a subspace
this decoder was never fitted to".

So refit.  For each arm this compares, on identical DEV rows and labels:

  observed_fit -> generated   the gate's number, reproduced
  generated_fit -> generated  fitted where it is evaluated
  generated_fit -> observed   cross-back: is the map merely rotated, or genuinely different?
  observed_fit -> observed    in-domain reference
  action_only / root_only     floors

Read-only over the published run.  It changes no gate output and authorizes nothing.
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
from d4mj.m03.gate import (OUTCOME_BINARY, STATIC_BINARY, STATIC_CONTINUOUS, M03Settings,
                           _binary_metrics, _fit_probe_many, _one_hot_actions, _regression_metrics)


def memory_report(features, split_rows, settings, device, *, targets, name, context):
    """The same refit question for the Mamba-state readouts.

    `h` needs no refit -- the memory panel already fits it on TRAIN `h`, which is why its
    observed and generated scores are identical. Only `z` and the joint `[z,h]` are fitted
    on observed successors and then read on generated ones, so only they can carry the
    artifact. `h` is therefore the control, not a condition.
    """
    train, dev = split_rows["train"], split_rows["dev"]
    key = "outcomes" if name == "outcomes" else "next_binary"
    train_truth = train[key].reshape(-1, len(targets)).float().to(device)
    dev_truth, roots = dev[key], dev["episode"]
    at = _one_hot_actions(len(train[key]), device=device)
    ad = _one_hot_actions(len(dev[key]), device=device)

    def flat(split, field, action):
        return torch.cat((features[split][field].flatten(0, 1).float().to(device), action), 1)

    gz, oz = f"{context}_generated_z", "observed_z"
    hh = f"{context}_next_h"
    train_obs_joint = torch.cat((features["train"][oz].flatten(0, 1).float().to(device),
                                 features["train"][hh].flatten(0, 1).float().to(device), at), 1)
    train_gen_joint = torch.cat((features["train"][gz].flatten(0, 1).float().to(device),
                                 features["train"][hh].flatten(0, 1).float().to(device), at), 1)
    dev_gen_joint = torch.cat((features["dev"][gz].flatten(0, 1).float().to(device),
                               features["dev"][hh].flatten(0, 1).float().to(device), ad), 1)
    out = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        z_obs = _fit_probe_many(flat("train", oz, at), train_truth,
                                {"gen": flat("dev", gz, ad)}, settings, hidden=hidden, binary=True)
        z_gen = _fit_probe_many(flat("train", gz, at), train_truth,
                                {"gen": flat("dev", gz, ad)}, settings, hidden=hidden, binary=True)
        j_obs = _fit_probe_many(train_obs_joint, train_truth, {"gen": dev_gen_joint},
                                settings, hidden=hidden, binary=True)
        j_gen = _fit_probe_many(train_gen_joint, train_truth, {"gen": dev_gen_joint},
                                settings, hidden=hidden, binary=True)
        h_fit = _fit_probe_many(flat("train", hh, at), train_truth,
                                {"gen": flat("dev", hh, ad)}, settings, hidden=hidden, binary=True)
        a_fit = _fit_probe_many(at, train_truth, {"gen": ad}, settings, hidden=hidden, binary=True)
        r_fit = _fit_probe_many(torch.cat((features["train"]["root_z"].float().to(device)
                                           .repeat_interleave(17, 0), at), 1), train_truth,
                                {"gen": torch.cat((features["dev"]["root_z"].float().to(device)
                                                   .repeat_interleave(17, 0), ad), 1)},
                                settings, hidden=hidden, binary=True)
        out[family] = {k: _binary_metrics(v["gen"], dev_truth, roots, targets, settings)
                       for k, v in (("z_observed_fit", z_obs), ("z_generated_fit", z_gen),
                                    ("joint_observed_fit", j_obs), ("joint_generated_fit", j_gen),
                                    ("h_native_fit", h_fit), ("action_only", a_fit),
                                    ("root_action", r_fit))}
    return out


def arm_report(features, split_rows, settings, device, *, targets, name):
    """One arm, one target family: every fit/evaluate combination on the same rows."""
    train, dev = split_rows["train"], split_rows["dev"]
    key = "outcomes" if name == "outcomes" else "next_binary"
    train_truth = train[key].reshape(-1, len(targets)).float().to(device)
    dev_truth = dev[key]
    roots = dev["episode"]
    action_train = _one_hot_actions(len(train[key]), device=device)
    action_dev = _one_hot_actions(len(dev[key]), device=device)

    def pack(tensor, action):
        return torch.cat((tensor.flatten(0, 1).float().to(device), action), 1)

    train_observed = pack(features["train"]["observed_successor"], action_train)
    train_generated = pack(features["train"]["generated_successor"], action_train)
    evaluation = {"generated": pack(features["dev"]["generated_successor"], action_dev),
                  "observed": pack(features["dev"]["observed_successor"], action_dev)}
    # The root latent repeated over the action fan-out: a persistence floor that knows the
    # present state and the action but nothing the world model predicted.
    root_train = torch.cat((features["train"]["projected"].float().to(device).repeat_interleave(17, 0),
                            action_train), 1)
    root_dev = torch.cat((features["dev"]["projected"].float().to(device).repeat_interleave(17, 0),
                          action_dev), 1)

    out = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        observed_fit = _fit_probe_many(train_observed, train_truth, evaluation, settings,
                                       hidden=hidden, binary=True)
        generated_fit = _fit_probe_many(train_generated, train_truth, evaluation, settings,
                                        hidden=hidden, binary=True)
        root_fit = _fit_probe_many(root_train, train_truth, {"dev": root_dev}, settings,
                                   hidden=hidden, binary=True)
        action_fit = _fit_probe_many(action_train, train_truth, {"dev": action_dev}, settings,
                                     hidden=hidden, binary=True)
        out[family] = {
            "observed_fit_on_generated": _binary_metrics(observed_fit["generated"], dev_truth, roots, targets, settings),
            "generated_fit_on_generated": _binary_metrics(generated_fit["generated"], dev_truth, roots, targets, settings),
            "generated_fit_on_observed": _binary_metrics(generated_fit["observed"], dev_truth, roots, targets, settings),
            "observed_fit_on_observed": _binary_metrics(observed_fit["observed"], dev_truth, roots, targets, settings),
            "root_action": _binary_metrics(root_fit["dev"], dev_truth, roots, targets, settings),
            "action_only": _binary_metrics(action_fit["dev"], dev_truth, roots, targets, settings),
        }
    return out


def continuous_report(features, split_rows, settings, device):
    """Same refit question against the continuous successor-state targets."""
    train, dev = split_rows["train"], split_rows["dev"]
    train_truth = train["next_continuous"].flatten(0, 1).float().to(device)
    dev_truth = dev["next_continuous"].flatten(0, 1).float()
    roots = dev["episode"]
    at = _one_hot_actions(len(train["next_continuous"]), device=device)
    ad = _one_hot_actions(len(dev["next_continuous"]), device=device)
    def pack(split, field, action):
        return torch.cat((features[split][field].flatten(0, 1).float().to(device), action), 1)
    fork_roots = roots.reshape(-1, 1).expand(len(roots), 17).reshape(-1)
    out = {}
    for hidden, family in ((False, "linear"), (True, "mlp")):
        obs = _fit_probe_many(pack("train", "observed_successor", at), train_truth,
                              {"gen": pack("dev", "generated_successor", ad)}, settings,
                              hidden=hidden, binary=False)
        gen = _fit_probe_many(pack("train", "generated_successor", at), train_truth,
                              {"gen": pack("dev", "generated_successor", ad)}, settings,
                              hidden=hidden, binary=False)
        out[family] = {k: _regression_metrics(v["gen"], dev_truth, fork_roots, STATIC_CONTINUOUS, settings)
                       for k, v in (("observed_fit", obs), ("generated_fit", gen))}
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_gates_20260916/m03_window")
    parser.add_argument("--memory-run", type=Path,
                        default=ROOT / "artifacts/lewm_gates_20260916/m03_window_memory")
    parser.add_argument("--contexts", default="c1,c4,c16,c64")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/lewm_gates_20260906/cache.sqlite3")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / "readout.json"
    if destination.exists():
        raise FileExistsError(f"generated_readout: refusing to replace {destination}")

    settings = M03Settings()
    sidecar = torch.load(args.run / "sidecar/sidecar.probe_only.pt", map_location="cpu", weights_only=False)
    cache = ArtifactCache(args.cache, settings=settings, device=args.device)
    report = {"schema": "d4mj_generated_readout_v1",
              "question": "is usable information absent from generated latents, or is the "
                          "observed-fit decoder failing on them?",
              "scope": "read-only refit over published M03 features; no gate output is changed",
              "run": str(args.run.resolve()),
              "sidecar_sha256": _sha256(args.run / "sidecar/sidecar.probe_only.pt"),
              "script_sha256": _sha256(Path(__file__)),
              "arms": {}, "m4_authorized": False}
    with use_cache(cache):
        for arm in ("raw", "tc", "direct_mamba", "direct_attention"):
            paths = {s: args.run / f"features/{arm}.{s}.pt" for s in ("train", "dev")}
            if not all(p.exists() for p in paths.values()):
                continue
            features = {s: resolve_payload(torch.load(p, map_location="cpu", weights_only=False))["features"]
                        for s, p in paths.items()}
            if "generated_successor" not in features["train"]:
                continue
            entry = {"feature_sha256": {s: _sha256(p) for s, p in paths.items()}}
            for name, targets in (("outcomes", OUTCOME_BINARY), ("successor_binary", STATIC_BINARY)):
                entry[name] = arm_report(features, sidecar["splits"], settings, args.device,
                                         targets=targets, name=name)
            entry["successor_continuous"] = continuous_report(features, sidecar["splits"], settings, args.device)
            report["arms"][arm] = entry
            print(json.dumps({"stage": "arm_complete", "arm": arm}), flush=True)
        # The Mamba-state readouts, same question, from the memory supplement's features.
        report["memory"] = {}
        for arm in ("raw", "tc"):
            paths = {s_: args.memory_run / f"features/{arm}.primary_{s_}.pt" for s_ in ("train", "dev")}
            if not all(q.exists() for q in paths.values()):
                continue
            mem = {s_: resolve_payload(torch.load(q, map_location="cpu", weights_only=False))["features"]
                   for s_, q in paths.items()}
            entry = {"feature_sha256": {s_: _sha256(q) for s_, q in paths.items()}, "contexts": {}}
            for ctx in args.contexts.split(","):
                if f"{ctx}_generated_z" not in mem["train"]:
                    continue
                entry["contexts"][ctx] = {
                    name: memory_report(mem, sidecar["splits"], settings, args.device,
                                        targets=targets, name=name, context=ctx)
                    for name, targets in (("outcomes", OUTCOME_BINARY), ("successor_binary", STATIC_BINARY))}
                print(json.dumps({"stage": "memory_context", "arm": arm, "context": ctx}), flush=True)
            report["memory"][arm] = entry
    atomic_manifest(destination, report)
    print(json.dumps({"status": "complete", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
