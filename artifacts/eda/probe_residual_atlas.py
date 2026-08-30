"""What is the remaining successor error, and is it a coverage problem?

Read-only. No head, loss or architecture change. The projection hypothesis is closed:
the null space carries damage-decodable structure but no correctable successor error.
This asks what the residual actually is.

    e = (z* - mean_a z*) - (zhat - mean_a zhat)

Part A archives two things that were previously asserted without an executable
artifact: the identical-successor aliasing count, and bootstrap intervals on the
residual R2 values, which were reported as point estimates.

Part B asks whether the error is coverage. For every test root, distance to its nearest
fit root in pooled and in hidden space, related to residual energy and NSE. The frozen
decoder's scaling rungs already showed action error falling monotonically with paired
roots -- 0.01742, 0.01667, 0.01615, 0.01521 over 456/912/1825/3651 -- without
saturating, so coverage is the standing suspect.

Part C stratifies residual energy by action identity, damaging against safe, identical
against distinct successor class, true effect magnitude, distinct-successor count and
termination, then correlates it with the simulator state already extracted for these
roots (health, inventory, tile window, mob geometry, and the never-rendered cooldown
and mob health).

Part D asks what the discarded null space encodes: a damage probe on the true successor
alone against one on [true successor, H_null]. If the null space adds nothing on top of
the target, its extra decodability is a redundant internal code and the drop across the
projection is benign compression.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_phase1b_fork import predict_branches
from probe_action_matching import classes_of
from probe_decoder_ceiling import cache as cache_backbone
from probe_hidden_ceiling import cache_hidden
from probe_hidden_forensics import fit_and_score, fit_residual, row_null_bases, standardise
from train_phase1b_fork import MixerWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config

DEVICE = "cuda"
SUFFIX = "abm0"


def aliasing(branch, labels):
    """Part A: do identical latent successors ever disagree on the damage label?"""
    eff = branch - branch.mean(1, keepdim=True)
    gram = torch.cdist(eff, eff).pow(2).numpy()
    classes = multi = disagree = 0
    for i in range(len(gram)):
        label = classes_of(gram[i])
        for k in range(label.max() + 1):
            members = label == k
            classes += 1
            if members.sum() > 1:
                multi += 1
                disagree += len(set(labels[i][members].tolist())) > 1
    return {"classes": classes, "multi_action_classes": multi, "label_disagreements": disagree}


def main() -> None:
    rows = load_forkset(HERE / f"forkset_{SUFFIX}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    masks = tuple(torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    terminated = torch.stack([r["terminated"] for r in rows]).numpy()
    keys = [(int(r["seed"]), int(r["step"])) for r in rows]

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = MixerWorld(config).to(DEVICE)
    load(HERE / f"phase1b_{SUFFIX}_n64" / "world_020000.pt", config, part0=world)
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)
    led = fork_actions(rows)

    result = {"arm": SUFFIX}
    result["aliasing"] = aliasing(branch, labels)
    a = result["aliasing"]
    print(f"Part A  aliasing: {a['classes']:,} classes, {a['multi_action_classes']:,} with "
          f">1 action, {a['label_disagreements']:,} disagreeing on the label", flush=True)

    predicted = predict_branches(world, config, history, led).float()
    target = ((branch - branch.mean(1, keepdim=True))
              - (predicted - predicted.mean(1, keepdim=True)))
    test = masks[2]
    energy = target[test].pow(2).sum(-1)                      # (n_test, 17)
    effect = (branch[test] - branch[test].mean(1, keepdim=True)).pow(2).sum(-1)
    print(f"        residual energy {float(energy.mean()):.4f} against action effect "
          f"{float(effect.mean()):.4f}  ratio {float(energy.sum()/effect.sum()):.4f}",
          flush=True)

    # ---------------------------------------------------------------- Part B: coverage
    pooled = cache_backbone(world, config, history, led, "pooled").flatten(1).float()
    hidden_raw = cache_hidden(world, config, history, led)
    with torch.no_grad():
        hidden = torch.cat([world.mix_norm(hidden_raw[lo:lo+64].to(DEVICE).float()).half().cpu()
                            for lo in range(0, len(hidden_raw), 64)])
    del hidden_raw
    fit_idx, test_idx = np.where(splits == "fit")[0], np.where(splits == "test")[0]
    coverage = {}
    for name, space in (("pooled", pooled),
                        ("hidden", hidden.float().mean(1).flatten(1))):
        f = torch.nn.functional.normalize(space[fit_idx], dim=1)
        t = torch.nn.functional.normalize(space[test_idx], dim=1)
        near = torch.cat([(1 - t[lo:lo+256] @ f.T).min(1).values
                          for lo in range(0, len(t), 256)])
        per_root = energy.mean(1)
        coverage[name] = {
            "corr_distance_energy": float(np.corrcoef(near.numpy(), per_root.numpy())[0, 1]),
            "quartiles": [float(near.quantile(q)) for q in (0.25, 0.5, 0.75)],
        }
        order = near.argsort()
        quarter = len(order) // 4
        coverage[name]["energy_by_density_quartile"] = [
            float(per_root[order[i * quarter:(i + 1) * quarter]].mean()) for i in range(4)]
        print(f"Part B  {name:<7} corr(nearest-fit distance, residual energy) "
              f"{coverage[name]['corr_distance_energy']:+.4f}   energy by closeness "
              f"quartile {['%.4f' % v for v in coverage[name]['energy_by_density_quartile']]}",
              flush=True)
    result["coverage"] = coverage

    # ---------------------------------------------------------------- Part C: atlas
    eff_test = branch[test] - branch[test].mean(1, keepdim=True)
    gram = torch.cdist(eff_test, eff_test).pow(2).numpy()
    n_classes = np.array([classes_of(gram[i]).max() + 1 for i in range(len(gram))])
    identical = np.zeros_like(labels[test.numpy()], dtype=bool)
    for i in range(len(gram)):
        label = classes_of(gram[i])
        sizes = np.bincount(label)
        identical[i] = sizes[label] > 1
    lab_t, term_t = labels[test.numpy()], terminated[test.numpy()]
    norm = energy / effect.clamp(min=1e-9)
    atlas = {
        "damaging": float(energy[torch.from_numpy(lab_t > 0)].mean()),
        "safe": float(energy[torch.from_numpy(lab_t <= 0)].mean()),
        "identical_class": float(energy[torch.from_numpy(identical)].mean()),
        "distinct_class": float(energy[torch.from_numpy(~identical)].mean()),
        "terminated": float(energy[torch.from_numpy(term_t > 0)].mean()),
        "not_terminated": float(energy[torch.from_numpy(term_t <= 0)].mean()),
        "corr_effect_magnitude": float(np.corrcoef(effect.flatten().numpy(),
                                                   energy.flatten().numpy())[0, 1]),
        "corr_n_classes": float(np.corrcoef(n_classes, energy.mean(1).numpy())[0, 1]),
        "by_action": {int(a): float(energy[:, a].mean()) for a in range(17)},
        "normalised_damaging": float(norm[torch.from_numpy(lab_t > 0)].mean()),
        "normalised_safe": float(norm[torch.from_numpy(lab_t <= 0)].mean()),
    }
    result["atlas"] = atlas
    print(f"Part C  energy  damaging {atlas['damaging']:.4f} vs safe {atlas['safe']:.4f}   "
          f"identical-class {atlas['identical_class']:.4f} vs distinct "
          f"{atlas['distinct_class']:.4f}")
    print(f"        terminated {atlas['terminated']:.4f} vs not {atlas['not_terminated']:.4f}"
          f"   corr(effect magnitude) {atlas['corr_effect_magnitude']:+.4f}"
          f"   corr(n classes) {atlas['corr_n_classes']:+.4f}")
    worst = sorted(atlas["by_action"].items(), key=lambda kv: -kv[1])[:5]
    print(f"        worst actions " + ", ".join(f"{a}:{v:.3f}" for a, v in worst), flush=True)

    # simulator state, joined by (seed, step)
    state = {(int(r["seed"]), int(r["step"])): r for r in
             torch.load(HERE / "state_features" / "features.pt", weights_only=False)}
    if all(k in state for k in keys):
        vis = torch.stack([state[k]["visible"] for k in keys]).float()
        hid = torch.stack([state[k]["hidden"] for k in keys]).float()
        both = torch.cat([vis, hid], dim=1)[test]
        per_root = energy.mean(1).numpy()
        corr = np.array([np.corrcoef(both[:, j].numpy(), per_root)[0, 1]
                         if both[:, j].std() > 0 else 0.0 for j in range(both.shape[1])])
        top = np.argsort(-np.abs(corr))[:8]
        result["state_correlations"] = {int(j): float(corr[j]) for j in top}
        print(f"        strongest state correlates with residual energy: " +
              ", ".join(f"dim{j}:{corr[j]:+.3f}" for j in top[:6]), flush=True)

    # ---------------------------------------------------------------- Part D: null content
    v_row, v_null = row_null_bases(world.readout.weight.detach().cpu())
    hc = hidden.float() - hidden.float().mean(1, keepdim=True)
    null = (hc @ v_null).reshape(*hc.shape[:2], -1).half()
    generator = torch.Generator().manual_seed(4)
    proj = torch.linalg.qr(torch.randn(null.shape[-1], 512, generator=generator))[0].half()
    true_eff = (branch - branch.mean(1, keepdim=True)).half()
    combos = {
        "true_successor": standardise(true_eff, masks[0]),
        "true_plus_null": torch.cat([standardise(true_eff, masks[0]),
                                     standardise(null @ proj, masks[0])], dim=-1),
    }
    result["null_content"] = {}
    for name, x in combos.items():
        result["null_content"][name] = fit_and_score(x, labels, masks, seed=11)
        r = result["null_content"][name]
        print(f"Part D  {name:<16} AUC {r['test_auc']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
              flush=True)

    (HERE / f"residual_atlas_{SUFFIX}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
