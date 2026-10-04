"""Post hoc to `geometry.py`: the cost of getting ONLY the consequence wrong, per state geometry.

geometry.py returned `per_tile_premise_fails` by its declared rule: the per-tile consequence share
(2.1%, codes 0.18%) is under 3x the pooled share (grid_pca 6.4%). Its direction also showed the
consequence is ONE tile, the health counter at (7,0) (94% of the direction). But its measure has a
confound I should have caught at design: the direction is the mean fatal-minus-surviving successor,
and fatal actions are mostly the stay-put ones, so in the pooled geometries it also picks up
stay-versus-move energy. You can see it in their lower validity (AUC 0.92-0.95 vs 0.998) and in
u_old's 28%, against the 0.04% DIAGNOSE.md measured along a discriminative direction. The pooled
baselines may be inflated; the declared reading cannot tell.

This measure has no direction and no fit. For each fatal branch, its TWIN is the same successor
frame with the health tile's pixels (rows 49-55, cols 0-6: `renderer.py:498-507`) copied from a
surviving branch at the same root. A frame and its twin differ only in the health counter, so
|x(frame) - x(twin)|^2 in a geometry is exactly what a world pays there for getting everything right
but the consequence. Relative to the mean within-root action effect per branch, that is the
consequence's weight in that geometry's loss. Damage is the same, with an unharmed surviving branch
as the donor.

Geometries as geometry.py, fitted the same way on FIT successors (grid PCA through the covariance,
K=512 codebook of layer-normed tokens, same seed); `grid` (pooled, no PCA) added. JUDGE roots only
are measured.

Reading, post hoc (committed before the run); it does not replace geometry.py's declared reading:
  cost(G) = mean over fatal branches |x - x_twin|^2 / mean over branches |x - root mean|^2
  per_tile_costlier   : some per-tile geometry (tokens_ln, codes) has cost >= 10x max(cost z, cost grid_pca)
  per_tile_not_costlier: every per-tile geometry has cost < 3x that
  mixed               : otherwise
"""

import argparse
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

from compactness import OLD, apply_pca, old_encoder  # noqa: E402
from confirm import seeds_for  # noqa: E402
from damage_direction import health_deltas  # noqa: E402
from geometry import HUD, K, TOKENS, WIDTH, encode, kmeans  # noqa: E402
from observability import load  # noqa: E402
from u_world import successors  # noqa: E402

HEALTH = (slice(49, 56), slice(0, 7))       # HUD row 7, column 0: token 63


def twins(frames, positive, donor_ok):
    """Each positive branch's frame with the health tile copied from the root's first donor branch."""
    donor = donor_ok.float().argmax(1)                                            # first eligible
    out = frames.clone()
    source = frames[torch.arange(len(frames)), donor]                             # [R, H, W, 3]
    patch = source[:, HEALTH[0], HEALTH[1]][:, None].expand(-1, frames.shape[1], -1, -1, -1)
    out[:, :, HEALTH[0], HEALTH[1]] = torch.where(positive[..., None, None, None], patch,
                                                  frames[:, :, HEALTH[0], HEALTH[1]])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--seed", type=int, default=20260925)
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
    checkpoint = args.run / "bridge/step-002000.pt"
    if _sha256(checkpoint) != json.loads((HERE / "evidence/confirm.json").read_text())["checkpoint_sha256"]:
        raise SystemExit("checkpoint differs from the one every earlier rung used")

    kept, sets = {}, {}
    for split in data:
        fatal = data[split]["p_death1"] > 0.5
        alive = ~fatal
        damaged = (dh[split][0] <= -2) & alive
        fatal_opp, damage_opp = fatal.any(1) & alive.any(1), damaged.any(1) & (~damaged & alive).any(1)
        keep = torch.where(fatal_opp | damage_opp)[0]
        kept[split] = succ[split][0][keep]
        sets[split] = {"fatal": (fatal[keep], torch.ones_like(fatal[keep]), alive[keep], fatal_opp[keep]),
                       "damage": (damaged[keep], alive[keep], ~damaged[keep] & alive[keep], damage_opp[keep])}
    del data, succ, dh

    from d4mj.experiments import _load_bridge_parent
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    del heads
    old = old_encoder(device)
    pca_old = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    fit = encode(bundle, old, pca_old, kept.pop("fit"), device)
    judge = {"frames": encode(bundle, old, pca_old, kept["judge"], device)}
    for outcome, (positive, valid, donor_ok, opp) in sets["judge"].items():
        rows = torch.where(opp)[0]
        judge[outcome] = (rows, encode(bundle, old, pca_old, twins(kept["judge"][rows], positive[rows],
                                                                   donor_ok[rows]), device))
    del bundle, old, kept
    torch.cuda.empty_cache()
    log(stage="encoded")

    # geometry.py's fitting, repeated exactly (same data, same seed).
    x = fit["grid"].flatten(0, 1)
    mean = x.mean(0, keepdim=True)
    cov = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64, device=device)
    for j in range(0, len(x), 8192):
        c = (x[j:j + 8192] - mean).to(device).double()
        cov += c.T @ c
    values, vectors = torch.linalg.eigh(cov / (len(x) - 1))
    order = values.argsort(descending=True)[:WIDTH]
    scale = values[order].clamp_min(0).sqrt()
    rank = int((scale > scale[0] * 1e-3).sum())
    pca = {"mean": mean.float(), "basis": vectors[:, order][:, :rank].float().cpu(), "rank": rank,
           "components": WIDTH}
    sample = fit["tokens"].flatten(0, 2)
    sample = sample[torch.randperm(len(sample), generator=torch.Generator().manual_seed(args.seed))[:400_000]]
    centroids = kmeans(nn.functional.layer_norm(sample.float(), (WIDTH,)), K, 30, args.seed, device)
    del fit, x, cov, vectors, sample

    def geometries(f):
        tokens = f["tokens"].float()
        ln = nn.functional.layer_norm(tokens, (WIDTH,))
        codes = torch.cat([torch.cdist(ln[j:j + 64].to(device).flatten(0, 2), centroids).argmin(1).cpu()
                           for j in range(0, len(ln), 64)]).reshape(ln.shape[:3])
        return {"z": f["z"], "grid": f["grid"], "grid_pca": apply_pca(pca, f["grid"]), "u_old": f["u_old"],
                "tokens": tokens, "tokens_ln": ln, "codes": nn.functional.one_hot(codes, K).float()}

    results = {}
    for outcome, (positive, valid, _, opp) in sets["judge"].items():
        rows, twin = judge[outcome]
        pos, val = positive[rows], valid[rows]
        acc = {}
        for j in range(0, len(rows), 64):                     # batched: one-hot codes are 41,472 wide
            a = geometries({k: v[rows[j:j + 64]] for k, v in judge["frames"].items()})
            b = geometries({k: v[j:j + 64] for k, v in twin.items()})
            p, w = pos[j:j + 64], val[j:j + 64].float()[..., None]
            for name in a:
                xa, xb = a[name].flatten(2), b[name].flatten(2)
                centre = (xa * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
                diff = ((xa - xb) ** 2)[p]
                s = acc.setdefault(name, {"effect": 0.0, "valid": 0, "diff": 0.0, "pos": 0, "tile": 0.0})
                s["effect"] += float((((xa - centre) * w) ** 2).sum())
                s["valid"] += int(w.sum())
                s["diff"] += float(diff.sum())
                s["pos"] += int(p.sum())
                if name in ("tokens", "tokens_ln", "codes"):
                    s["tile"] = s["tile"] + diff.reshape(len(diff), TOKENS, -1).sum((0, 2))
        results[outcome] = {"roots": len(rows), "branches": int(pos.sum())}
        for name, s in acc.items():
            effect, consequence = s["effect"] / s["valid"], s["diff"] / s["pos"]
            r = {"cost": consequence / effect, "effect_per_branch": effect, "consequence_per_branch": consequence}
            if name in ("tokens", "tokens_ln", "codes"):
                r["share_on_health_tile"] = float(s["tile"][HUD] / s["tile"].sum())
            results[outcome][name] = r
            log(outcome=outcome, geometry=name, cost=f'{r["cost"]:.3e}')

    c = results["fatal"]
    base = max(c["z"]["cost"], c["grid_pca"]["cost"])
    tiles = [c["tokens_ln"]["cost"], c["codes"]["cost"]]
    reading = ("per_tile_costlier" if max(tiles) >= 10 * base else
               "per_tile_not_costlier" if max(tiles) < 3 * base else "mixed")
    evidence = {"schema": "d4mj_twin_cost_v1", "status": "POST HOC, EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "identity": recorded,
                "checkpoint_sha256": _sha256(checkpoint), "reading": reading, "result": results}
    (args.out / "twin.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="twin_complete", reading=reading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
