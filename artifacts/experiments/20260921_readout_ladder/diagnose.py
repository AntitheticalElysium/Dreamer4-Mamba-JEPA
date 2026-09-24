"""Why does the u->u world's prediction lose the mob consequences its input carries?

U_WORLD.md: root u 0.804, generated u 0.727 (-0.078, zombie -0.100). The loss is located in the
predicted state; its cause is not. A one-day diagnostic queue, each part localization, none a proof:

  critical   Action-critical prediction error. Fit the fatal-versus-safe direction w on
             WITHIN-ROOT-CENTRED real successor u, FIT roots only; require that it reads held-out
             real successors (within-root AUC >= 0.9) or the part is void. Then, on judgement
             opportunity roots, compare the world's within-root error along w -- normalized by the
             true action effect along w -- with its average normalized within-root error, and
             report how much of the true action effect along w the world reproduces. A
             copy-the-root baseline calibrates both. Zombie / lava / night strata.
             This is where the September 19 report once leapt from "small energy share" to "the
             loss under-allocates" and had to retract it: nothing here licenses that leap.

  exposure   Training exposure of the old u->u world. Rebuilds its exact 25,600-window pool by
             replaying its sampler (verified against the cached action arrays, byte for byte), labels
             every unique (episode, step) transition from what the corpus stores -- damage from the
             reward (achievements are integers, health is paid at 0.1 per point), death from the
             terminal flag, zombie adjacency and night from pixels by classifiers trained on the FIT
             fork roots, whose simulator state is known, and validated on the fresh roots -- and
             stratifies the world's one-step prediction error on those logged transitions, overall
             and along w. The same exposure counts are taken over the M4 training corpus.

Declared reading, fixed before either part runs:
  sparse_exposure       fewer than 200 unique near-zombie transitions with a stay-type action, or
                        fewer than 200 with a movement action, or fewer than 50 unique near-zombie
                        transitions followed by mob-scale damage (health -2 or worse), in the pool
  concentrated_error    within-root normalized error along w more than twice the average
                        normalized within-root error, with the direction valid
  -> sparse_exposure                          points at logged hazard data
  -> adequate exposure, concentrated_error    points at the prediction loss
  -> adequate exposure, error not concentrated is the ambiguous case (averaging or dynamics):
                                              ONLY THEN does part 3, the averaged-successor test, run
  -> the direction invalid                    part 3 runs too, since part 2 decided nothing
The thresholds are judgment calls, stated here so they cannot move after the numbers.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE

from compactness import OLD, OLD_CHECKPOINT  # noqa: E402
from confirm import seeds_for  # noqa: E402
from frozen_ladder import strata  # noqa: E402
from observability import load  # noqa: E402

N = 17
MOVES = (1, 2, 3, 4)                    # LEFT RIGHT UP DOWN; everything else leaves the player in place
TILE, ROWS, COLS = 7, 7, 9
NEIGHBOURS = ((2, 4), (4, 4), (3, 3), (3, 5))


def within_auc(score, fatal):
    """Mean over roots with both classes of P(score_fatal > score_safe)."""
    out = []
    for s, f in zip(score, fatal):
        if f.any() and (~f).any():
            a, b = s[f][:, None], s[~f][None, :]
            out.append(float((a > b).float().mean() + 0.5 * (a == b).float().mean()))
    return float(np.mean(out)) if out else float("nan")


def critical(data, inner):
    """The action-critical error, from real and generated successor u on the observability roots."""
    fit_rows = torch.where(~inner)[0]
    real_f, fatal_f = data["fit"]["real_u"][fit_rows], data["fit"]["p_death1"][fit_rows] > 0.5
    opp_f = fatal_f.any(1) & (~fatal_f).any(1)
    centre = lambda x: x - x.mean(1, keepdim=True)
    xc = centre(real_f[opp_f]).reshape(-1, real_f.shape[-1])
    y = fatal_f[opp_f].reshape(-1).float()
    scale = xc.std(0).clamp_min(1e-6)
    torch.manual_seed(0)
    probe = nn.Linear(xc.shape[-1], 1)
    opt = torch.optim.LBFGS(probe.parameters(), max_iter=500, line_search_fn="strong_wolfe")
    weight = (1 - y).sum() / y.sum().clamp_min(1)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(probe(xc / scale)[:, 0], y, pos_weight=weight) \
            + 1e-3 * probe.weight.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    w = (probe.weight.detach()[0] / scale)
    w = w / w.norm()

    judge = data["judge"]
    fatal = judge["p_death1"] > 0.5
    opp = fatal.any(1) & (~fatal).any(1)
    real, gen, root = judge["real_u"], judge["generated_u"], judge["root_u"]
    copy = root[:, None].expand_as(real)
    out = {"direction_heldout_within_root_auc": within_auc((centre(real) @ w)[opp], fatal[opp]),
           "direction_heldout_auc_on_generated": within_auc((centre(gen) @ w)[opp], fatal[opp])}
    strat = strata(judge["visible"])

    def measure(mask):
        rc, gc, cc = centre(real[mask]), centre(gen[mask]), centre(copy[mask])
        effect_w = (rc @ w).square().mean()
        effect_all = rc.square().sum(-1).mean()
        block = {"roots": int(mask.sum()),
                 "within_root_energy_share_of_w": float(effect_w / effect_all),
                 "world": {"normalized_error_all": float((gc - rc).square().sum(-1).mean() / effect_all),
                           "normalized_error_w": float(((gc - rc) @ w).square().mean() / effect_w),
                           "effect_ratio_w": float(((gc @ w).square().mean() / effect_w).sqrt()),
                           "effect_correlation_w": float(torch.corrcoef(torch.stack(((gc @ w).reshape(-1), (rc @ w).reshape(-1))))[0, 1])},
                 "copy_root": {"normalized_error_all": float((cc - rc).square().sum(-1).mean() / effect_all),
                               "normalized_error_w": float(((cc - rc) @ w).square().mean() / effect_w)},
                 "absolute_error_all": float((gen[mask] - real[mask]).square().sum(-1).mean()),
                 "absolute_error_w": float(((gen[mask] - real[mask]) @ w).square().mean())}
        block["world"]["ratio_w_to_all"] = block["world"]["normalized_error_w"] / block["world"]["normalized_error_all"]
        return block

    out["all_opportunity"] = measure(opp)
    for name in ("zombie_adjacent", "lava_adjacent", "night", "day"):
        out[name] = measure(opp & strat[name])
    out["valid"] = bool(out["direction_heldout_within_root_auc"] >= 0.9)
    out["concentrated_error"] = bool(out["valid"] and out["all_opportunity"]["world"]["ratio_w_to_all"] > 2.0)
    return out, w


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("critical",), required=True)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")

    from u_world import WORLD, successors, u_world_features
    from compactness import old_encoder
    from d4mj.config import config_from_dict
    from d4mj.lewm import LeWMWorld
    fit_seeds, _ = seeds_for(json.loads(args.partition.read_text()), FORK_STORE)
    data = load(fit_seeds)
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if {s: data[s]["identity"] for s in data} != recorded:
        raise SystemExit("data differs from what the observability test measured")
    succ = successors(fit_seeds)
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
    groups = data["fit"]["seed"].unique()
    held = set(groups[torch.randperm(len(groups), generator=torch.Generator().manual_seed(args.seed))
                      [:max(1, round(0.15 * len(groups)))]].tolist())
    inner = torch.tensor([int(s) in held for s in data["fit"]["seed"]])
    log(stage="features")
    result, w = critical(data, inner)
    torch.save({"w": w}, args.out / "critical_direction.pt")
    evidence = {"schema": "d4mj_diagnose_critical_v1", "status": "EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "world_sha256": _sha256(WORLD),
                "direction_sha256": _sha256(args.out / "critical_direction.pt"), "identity": recorded,
                "result": result}
    (args.out / "diagnose_critical.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="critical_complete", valid=result["valid"], concentrated=result["concentrated_error"],
        auc=round(result["direction_heldout_within_root_auc"], 4),
        ratio=round(result["all_opportunity"]["world"]["ratio_w_to_all"], 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
