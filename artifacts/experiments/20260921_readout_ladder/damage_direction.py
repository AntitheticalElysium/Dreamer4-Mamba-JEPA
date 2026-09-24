"""Post-hoc discriminator: never shown a death, or cannot learn a small direction even when shown?

The declared queue (`diagnose.py`) returned: direction valid, error concentrated along the fatal
direction (ratio 25), exposure not sparse by its thresholds -- which reads "the prediction loss".
The exposure audit then found what those thresholds did not count: the old u->u world's training
pool holds ZERO death transitions. Its TC-consecutive window spans 13 native frames and predicts only
the first three transitions (`JointSampler`: starts in [0, len + 1 - span]), so the last 9 steps of
every episode, every death included, can never be a target. The fatal direction is the look of a dead
successor, which this world never saw. Two explanations now fit and the declared queue cannot part
them; the averaged-successor test (part 3, not triggered by its declared condition) would not either.

The pool DOES contain non-fatal damage: 894 transitions losing 2+ health, 694 of them beside a zombie.
So fit a DAMAGE direction the same way as the fatal one -- surviving branches only, damaged (health
-2 or worse) versus unharmed, within-root centred over the surviving branches, FIT roots only -- and
ask whether the world reproduces an action consequence it WAS trained on.

Declared reading, committed before the run:
  the damage direction reads held-out real successors below within-root AUC 0.9  -> void
  generated within-root AUC along it >= 0.75 and error ratio <= 2   -> death_unseen_damage_learned:
        the world learns a consequence it is shown; the fatal failure is the missing outcome
  generated within-root AUC < 0.60 and error ratio > 2              -> loss_fails_with_exposure:
        it fails even a consequence it was shown; the loss (or dynamics) is implicated
  otherwise                                                        -> mixed
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from compactness import OLD, OLD_CHECKPOINT, old_encoder  # noqa: E402
from confirm import seeds_for  # noqa: E402
from diagnose import within_auc  # noqa: E402
from frozen_ladder import strata  # noqa: E402
from observability import FIT_STATE, FRESH, load  # noqa: E402
from u_world import WORLD, successors, u_world_features  # noqa: E402


def health_deltas(fit_seeds):
    """Per-action health change, in exactly `observability.load`'s row order; identity re-derived."""
    store = {int(p.stem.split("-")[1]): p for p in FORK_STORE.glob("seed-*.pt")}
    state = {int(p.stem.split("-")[1]): p for p in FIT_STATE.glob("seed-*.pt")}
    out = {}
    for split, files in (("fit", [(store[s], state[s]) for s in sorted(fit_seeds)]),
                         ("judge", [(p, None) for p in sorted(FRESH.glob("seed-*.pt"))])):
        deltas, identity = [], []
        for pixel_file, state_file in files:
            pixels = {int(r["step"]): r for r in torch.load(pixel_file, weights_only=False)}
            order = pixels if state_file is None else {int(r["step"]): r for r in torch.load(state_file, weights_only=False)}
            for step, f in order.items():
                deltas.append(pixels[step]["health_delta"].float())
                identity.append((int(f["seed"]), int(pixels[step]["frames"].sum())))
        out[split] = (torch.stack(deltas), hashlib.sha256(repr(identity).encode()).hexdigest())
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded:
        raise SystemExit("data differs from what the observability test measured")
    succ, dh = successors(fit_seeds), health_deltas(fit_seeds)
    if {s: succ[s][1] for s in succ} != recorded or {s: dh[s][1] for s in dh} != recorded:
        raise SystemExit("successor or health rows are not aligned with the measured rows")

    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMWorld
    stored = torch.load(OLD_CHECKPOINT, map_location="cpu", weights_only=False)
    world = LeWMWorld(config_from_dict(stored["config"])).to(device)
    del stored
    world.load_state_dict(torch.load(WORLD, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    world.eval()
    encoder = old_encoder(device)
    pca = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    for split in data:
        data[split].update(u_world_features(encoder, world, pca, data[split]["frames"], data[split]["actions"],
                                            succ[split][0], device))
        data[split]["health_delta"] = dh[split][0]
    log(stage="features")

    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])

    def labels(d):
        alive = d["p_death1"] < 0.5
        damaged = (d["health_delta"] <= -2) & alive
        opp = (damaged.any(1)) & ((~damaged & alive).any(1))
        return alive, damaged, opp

    def centre(x, alive):
        m = alive.float()[..., None]
        return (x - (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)) * m

    fit = {k: v[~inner] for k, v in data["fit"].items() if torch.is_tensor(v)}
    alive_f, dmg_f, opp_f = labels(fit)
    xc = centre(fit["real_u"], alive_f)[opp_f][alive_f[opp_f]]
    y = dmg_f[opp_f][alive_f[opp_f]].float()
    scale = xc.std(0).clamp_min(1e-6)
    torch.manual_seed(0)
    probe = nn.Linear(xc.shape[-1], 1)
    opt = torch.optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    pos = (1 - y).sum() / y.sum().clamp_min(1)

    def closure():
        opt.zero_grad()
        l = nn.functional.binary_cross_entropy_with_logits(probe(xc / scale)[:, 0], y, pos_weight=pos) \
            + 1e-3 * probe.weight.square().sum()
        l.backward()
        return l
    opt.step(closure)
    w = probe.weight.detach()[0] / scale
    w = w / w.norm()

    judge = data["judge"]
    alive, damaged, opp = labels(judge)
    real, gen = centre(judge["real_u"], alive), centre(judge["generated_u"], alive)
    masked = lambda s: torch.where(alive, s, torch.full_like(s, float("nan")))

    def auc(x, mask):
        scores = masked(x @ w)[mask]
        return within_auc([s[~s.isnan()] for s in scores], [d[a] for d, a in zip(damaged[mask], alive[mask])])

    strat = strata(judge["visible"])

    def measure(mask):
        a = alive[mask]
        rc, gc = real[mask][a], gen[mask][a]
        effect_w, effect_all = (rc @ w).square().mean(), rc.square().sum(-1).mean()
        ne_all = float((gc - rc).square().sum(-1).mean() / effect_all)
        ne_w = float(((gc - rc) @ w).square().mean() / effect_w)
        return {"roots": int(mask.sum()), "real_within_root_auc": auc(judge["real_u"], mask),
                "generated_within_root_auc": auc(judge["generated_u"], mask),
                "energy_share_of_w": float(effect_w / effect_all), "normalized_error_all": ne_all,
                "normalized_error_w": ne_w, "ratio_w_to_all": ne_w / ne_all,
                "effect_ratio_w": float(((gc @ w).square().mean() / effect_w).sqrt()),
                "effect_correlation_w": float(torch.corrcoef(torch.stack((gc @ w, rc @ w)))[0, 1])}

    result = {"damage_opportunity_roots": {"fit": int(opp_f.sum()), "judge": int(opp.sum())},
              "all": measure(opp), "zombie_adjacent": measure(opp & strat["zombie_adjacent"]),
              "night": measure(opp & strat["night"])}
    a = result["all"]
    reading = ("void" if a["real_within_root_auc"] < 0.9 else
               "death_unseen_damage_learned" if a["generated_within_root_auc"] >= 0.75 and a["ratio_w_to_all"] <= 2 else
               "loss_fails_with_exposure" if a["generated_within_root_auc"] < 0.60 and a["ratio_w_to_all"] > 2 else
               "mixed")
    evidence = {"schema": "d4mj_damage_direction_v1", "status": "POST HOC, EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "world_sha256": _sha256(WORLD), "identity": recorded,
                "reading": reading, "result": result}
    (args.out / "damage_direction.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="damage_direction_complete", reading=reading,
        real_auc=round(a["real_within_root_auc"], 4), generated_auc=round(a["generated_within_root_auc"], 4),
        ratio=round(a["ratio_w_to_all"], 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
