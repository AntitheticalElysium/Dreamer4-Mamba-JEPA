"""Where does the u->u transition lose the zombie consequence? DIAGNOSE.md's fatal-direction test, applied
to the interface worlds (U and Z, seeds 1 and 2).

Replication (INTERFACE.md): U's imagined state loses ~5 points against its own root on zombie roots, and
the loss sits in the transition (root_U ~ patch tokens there; generated_U -0.07* below them). DIAGNOSE.md
found the old u->u world reproduced 95.5% of each action's effect but none of the fatal direction (ratio
25, generated AUC 0.495). Does the new world fail the same way, how much less, and is the failure
misalignment or shrinkage?

POST HOC: FIT roots fit the directions; the 56k block (read twice) judges. Per world:
  fatal direction   logistic on within-root-centred REAL successor states of FIT opportunity roots
                    (32-key P(death1) > 0.5), as diagnose.py
  damage direction  surviving branches, health -2 or worse vs unharmed, centred over survivors, as
                    damage_direction.py
Measured on judge opportunity roots (and zombie-adjacent ones):
  real / generated within-root AUC along the direction
  normalized error overall and along the direction (effect energy as denominator), and their ratio
  generated/real effect magnitude along the direction, and their correlation
  share of the within-root effect energy along the direction, plain and under the world's own loss weights
  per-variance-bin effect R^2 (the world's coordinates grouped by TRAIN variance rank)
Readings (descriptive, committed before the run), per world and direction:
  real AUC < 0.9                          -> void
  generated AUC >= 0.9                    -> consequence_transferred
  ratio > 2 and effect_corr < 0.3         -> misaligned (the world moves along it, but not like reality)
  ratio > 2 and effect_ratio < 0.5        -> shrunk (the world barely moves along it)
  otherwise                               -> partial
"""

import json
import os
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

N = 17
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"
WORLDS = {"U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"), "U_s2": ("U", ROOT / "artifacts/eda/interface_worlds_v2/U.pt"),
          "Z_s1": ("Z", ROOT / "artifacts/eda/interface_worlds_v1/Z.pt"), "Z_s2": ("Z", ROOT / "artifacts/eda/interface_worlds_v2/Z.pt")}
BINS = ((0, 10), (10, 30), (30, 60), (60, 100), (100, 192))


def direction(x, y, seed=0):
    """Unit direction (in raw coordinates) of a class-balanced logistic probe on standardized x."""
    from torch import optim
    scale = x.std(0).clamp_min(1e-6)
    torch.manual_seed(seed)
    probe = nn.Linear(x.shape[-1], 1)
    opt = optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    pos = (1 - y).sum() / y.sum().clamp_min(1)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(probe(x / scale)[:, 0], y, pos_weight=pos) \
            + 1e-3 * probe.weight.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    w = probe.weight.detach()[0] / scale
    return w / w.norm()


def centre(x, keep):
    m = keep.float()[..., None]
    return (x - (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)) * m


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from damage_direction import health_deltas
    from diagnose import within_auc
    from frozen_ladder import strata
    from interface import POOL, branches, encode, load_bridge, state_of, world_bundle
    from observability import load
    from u_world import successors

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    succ, dh = successors(fit_seeds), health_deltas(fit_seeds)
    if data["fit"]["identity"] != recorded["fit"] or succ["fit"][1] != recorded["fit"] or dh["fit"][1] != recorded["fit"]:
        raise SystemExit("FIT rows misaligned")
    fit = data["fit"]
    fit["successors"], fit["health_delta"] = succ["fit"][0], dh["fit"][0]
    del data, succ
    judge, manifest, _ = judge_store(JUDGE)
    rows = [r for f in sorted(JUDGE.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    judge["successors"] = torch.stack([r["successors"] for r in rows])
    judge["health_delta"] = torch.stack([r["health_delta"] for r in rows]).float()
    del rows
    encoder, config = load_bridge()

    def real_states(arm, frames):
        out = []
        for i in range(0, len(frames), 16):
            z, grid = encode(encoder, frames[i:i + 16].flatten(0, 1)[:, None], device)
            out.append(state_of(arm, pool["pca"], z, grid)[:, 0].view(-1, N, 192))
        return torch.cat(out)

    real = {arm: {s: real_states(arm, d["successors"]) for s, d in (("fit", fit), ("judge", judge))} for arm in ("U", "Z")}
    log(stage="real_states")
    strat = strata(judge["visible"])
    result = {}
    for name, (arm, path) in WORLDS.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        bundle = world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device).eval()
        gen = {s: branches(bundle, heads, pool["pca"], arm, encoder, d["frames"], d["actions"], device)["generated"][..., :192]
               for s, d in (("fit", fit), ("judge", judge))}
        del bundle, heads
        lam = pool["weights"][arm]
        target_var = pool[{"U": "u", "Z": "z"}[arm]][~pool["terminal"]][:, 1:].reshape(-1, 192).var(0)
        order = target_var.argsort(descending=True)
        result[name] = {}
        for outcome in ("fatal", "damage"):
            def labels(d):
                p = d["p_death1"]
                alive = p <= 0.5
                if outcome == "fatal":
                    pos, keep = p > 0.5, torch.ones_like(alive)
                else:
                    pos, keep = (d["health_delta"] <= -2) & alive, alive
                opp = (pos & keep).any(1) & (~pos & keep).any(1)
                return pos, keep, opp
            pos_f, keep_f, opp_f = labels(fit)
            xf = centre(real[arm]["fit"], keep_f)[opp_f][keep_f[opp_f]]
            w = direction(xf, pos_f[opp_f][keep_f[opp_f]].float())
            pos, keep, opp = labels(judge)
            rc, gc = centre(real[arm]["judge"], keep), centre(gen["judge"], keep)
            block = {}
            for sname, mask in (("all", opp), ("zombie", opp & strat["zombie_adjacent"])):
                k = keep[mask]
                r, gg = rc[mask][k], gc[mask][k]
                sc = lambda x: [s[kk] for s, kk in zip((x[mask] @ w), keep[mask])]
                energy_w, energy_all = (r @ w).square().mean(), r.square().sum(-1).mean()
                ne_all = float((gg - r).square().sum(-1).mean() / energy_all)
                ne_w = float(((gg - r) @ w).square().mean() / energy_w)
                bins = {}
                for lo, hi in BINS:
                    idx = order[lo:hi]
                    bins[f"{lo}-{hi}"] = {"effect_r2": float(1 - (gg[:, idx] - r[:, idx]).square().sum() / r[:, idx].square().sum()),
                                          "w_share": float(w[idx].square().sum()),
                                          "effect_share": float(r[:, idx].square().sum() / r.square().sum())}
                block[sname] = {"roots": int(mask.sum()),
                                "real_auc": within_auc(sc(real[arm]["judge"]), [pp[kk] for pp, kk in zip(pos[mask], keep[mask])]),
                                "generated_auc": within_auc(sc(gen["judge"]), [pp[kk] for pp, kk in zip(pos[mask], keep[mask])]),
                                "normalized_error_all": ne_all, "normalized_error_w": ne_w, "ratio": ne_w / ne_all,
                                "effect_ratio_w": float(((gg @ w).square().mean() / energy_w).sqrt()),
                                "effect_corr_w": float(torch.corrcoef(torch.stack((gg @ w, r @ w)))[0, 1]),
                                "energy_share_w": float(energy_w / energy_all),
                                "weighted_share_w": float(((r * lam.sqrt()) @ ((w * lam.sqrt()) / (w * lam.sqrt()).norm())).square().mean()
                                                          / (r.square() * lam).sum(-1).mean()),
                                "effect_r2_all": 1 - ne_all, "bins": bins}
            a = block["all"]
            block["reading"] = ("void" if a["real_auc"] < 0.9 else "consequence_transferred" if a["generated_auc"] >= 0.9 else
                                "misaligned" if a["ratio"] > 2 and a["effect_corr_w"] < 0.3 else
                                "shrunk" if a["ratio"] > 2 and a["effect_ratio_w"] < 0.5 else "partial")
            result[name][outcome] = block
            log(world=name, outcome=outcome, reading=block["reading"],
                **{k: round(v, 4) for k, v in a.items() if isinstance(v, float)},
                zombie_gen_auc=round(block["zombie"]["generated_auc"], 4), zombie_real_auc=round(block["zombie"]["real_auc"], 4))
    evidence = {"schema": "d4mj_transition_diag_v1", "status": "POST HOC: FIT fits, the 56k block (read twice) judges",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "result": result}
    (HERE / "evidence/transition_diag.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="transition_diag_complete")


if __name__ == "__main__":
    main()
