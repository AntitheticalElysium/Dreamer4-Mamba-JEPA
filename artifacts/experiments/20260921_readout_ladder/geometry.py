"""Premise check for a per-tile world: where does the survival consequence live, and how much of a
world's prediction loss would it carry, under each candidate state geometry?

A/B/C (ABC.md) returned `stop_loss_tweaking`: go to spatial dynamics or uncertainty modelling. The
outside literature that solved this class of failure -- Dedieu et al. 2025 on Craftax-Classic itself
(the same 63x63 frames, 9x9 tiles of 7x7 pixels), EMERALD on Crafter, DINO-WM, V-JEPA 2-AC -- gives
the world one token per tile and a per-token loss. Their argument, applied to us: the consequence of
an action is a change to ONE HUD tile (the health counter; fork frames are rendered with
auto_reset=False, so a fatal successor shows health 0, not a reset), so a per-tile loss gives it a
term of its own, where a pooled global vector (CLS -> z, or the 4x4-pooled grid -> PCA u) makes it a
tiny, low-variance component that uniform MSE neglects (DIAGNOSE.md: 0.04% of the effect energy).
That premise is measured here, before anything is built. Nothing is trained.

Data: the observability roots (EXPLORATORY), pinned by content hash; every root with both a fatal and
a surviving action (fatal: 32-key P(death1) > 0.5, as diagnose.py) or with both a damaged and an
unharmed surviving action (damage: health -2 or worse, as damage_direction.py). All 17 REAL
successor frames, encoded by the frozen canonical Raw H2 encoder (the checkpoint every rung used),
plus the old TC encoder's u as the anchor to DIAGNOSE.md.

Geometries -- each is the space a world's loss would be computed in:
  z          projected CLS, what canonical LeWM's world predicts (MSE)
  grid_pca   4x4-pooled patch grid -> PCA-192 fit on FIT successors (u's construction, Raw H2)
  u_old      the old u->u world's own state: TC encoder, persisted PCA (anchor)
  tokens     the 81 per-tile output tokens, flattened (per-token MSE)
  tokens_ln  the same, each token layer-normalized (V-JEPA 2-AC's `normalize_reps`)
  codes      each tile's nearest of K=512 k-means centroids of FIT tokens_ln, one-hot (per-tile CE:
             squared one-hot distance = 2 x flipped tiles)

Measure, one method for all geometries, no probe to overfit 15,552 dimensions: the consequence
direction d = mean over FIT roots of (mean fatal successor - mean surviving successor) at that root.
Root-specific differences (which way the view scrolls) average out; what is consistent across roots
stays. On JUDGE roots:
  validity  within-root AUC of x . d over real successors
  share     energy of the within-root action effect along d / total within-root effect energy
            (the fraction of a world's loss, in that geometry, that the consequence carries)
  locality  for per-tile geometries: fewest tile positions holding half of |d|^2, and which
Damage is measured the same way over surviving branches (centred over them), reported, not ruled.

Declared reading, fatal consequence (committed before the run):
  a per-tile geometry (tokens_ln or codes) is VALID if its judge AUC >= 0.90
  per_tile_premise_holds  : some valid per-tile geometry has share >= 10x the largest share among
                            the valid pooled geometries (z, grid_pca), AND locality <= 3 tiles
  per_tile_premise_fails  : every valid per-tile geometry has share < 3x that pooled share, OR
                            locality > 10 tiles
  void                    : no per-tile geometry is valid, or no pooled geometry is valid
  mixed                   : otherwise
"""

import argparse
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

from compactness import OLD, apply_pca, fit_pca, old_encoder  # noqa: E402
from confirm import seeds_for  # noqa: E402
from damage_direction import health_deltas  # noqa: E402
from diagnose import within_auc  # noqa: E402
from observability import load  # noqa: E402
from u_world import successors  # noqa: E402

N, TOKENS, WIDTH, K = 17, 81, 192, 512
HUD = 63          # tokens 63-80 are the two HUD rows (7x9 map above them)


@torch.no_grad()
def encode(bundle, old, pca_old, succ, device, batch=8):
    """Every geometry's raw material for [roots, 17] real successors."""
    out = {k: [] for k in ("z", "grid", "tokens", "u_old")}
    for i in range(0, len(succ), batch):
        f = succ[i:i + batch].to(device)
        n = len(f)
        z, _, tokens, _, _ = bundle.encoder._hidden(f)
        grid = nn.functional.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(-1, WIDTH, 9, 9), 4).flatten(1)
        _, _, g = old.export(f.flatten(0, 1).unsqueeze(1), grid=4)
        out["z"].append(z.reshape(n, N, -1).cpu())
        out["grid"].append(grid.reshape(n, N, -1).cpu())
        out["tokens"].append(tokens.reshape(n, N, TOKENS, WIDTH).half().cpu())
        out["u_old"].append(apply_pca(pca_old, g.flatten(2).cpu())[:, 0].reshape(n, N, -1))
    return {k: torch.cat(v) for k, v in out.items()}


@torch.no_grad()
def kmeans(x, k, iters, seed, device):
    g = torch.Generator().manual_seed(seed)
    c = x[torch.randperm(len(x), generator=g)[:k]].to(device).float()
    x = x.to(device).float()
    for _ in range(iters):
        assign = torch.cat([torch.cdist(x[j:j + 65536], c).argmin(1) for j in range(0, len(x), 65536)])
        sums = torch.zeros_like(c).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).float()
        c = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], c)
    return c


def measure(get, rows, labels, device, width_per_tile=None, batch=32):
    """d from FIT, then share, AUC and locality on JUDGE, for one outcome (fatal or damage)."""
    def pieces(split, idx):
        x = get(split, idx).to(device).float()                              # [b, 17, d]
        pos, valid = (t[idx].to(device) for t in labels[split])             # outcome, branches counted
        v = valid.float()[..., None]
        mean = lambda m: (x * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)
        return x, pos, valid, v, mean

    total, count = None, 0
    for j in range(0, len(rows["fit"]), batch):
        x, pos, valid, v, mean = pieces("fit", rows["fit"][j:j + batch])
        d = mean((pos & valid).float()) - mean((~pos & valid).float())
        total = d.sum(0) if total is None else total + d.sum(0)
        count += len(d)
    d = total / count
    unit = d / d.norm()

    along, energy, per_tile, scores, outcome = 0.0, 0.0, None, [], []
    for j in range(0, len(rows["judge"]), batch):
        x, pos, valid, v, mean = pieces("judge", rows["judge"][j:j + batch])
        xc = (x - mean(valid.float())[:, None]) * v
        along += float(((xc @ unit) ** 2 * valid).sum())
        energy += float((xc ** 2).sum())
        if width_per_tile:
            tile = (xc.reshape(*xc.shape[:2], TOKENS, width_per_tile) ** 2).sum((0, 1, 3)).cpu()
            per_tile = tile if per_tile is None else per_tile + tile
        s = (x @ unit).cpu()
        for r in range(len(x)):
            keep = valid[r].cpu()
            scores.append(s[r][keep])
            outcome.append(pos[r].cpu()[keep])
    result = {"judge_within_root_auc": within_auc(scores, outcome), "share": along / energy,
              "roots": {"fit": len(rows["fit"]), "judge": len(rows["judge"])}}
    if width_per_tile:
        mass = (d.reshape(TOKENS, width_per_tile) ** 2).sum(-1).cpu()
        order = mass.argsort(descending=True)
        cumulative = mass[order].cumsum(0) / mass.sum()
        result.update({
            "locality_tiles_for_half": int((cumulative < 0.5).sum()) + 1,
            "top_tiles": [{"tile": int(t), "row": int(t) // 9, "col": int(t) % 9,
                           "share_of_direction": float(mass[t] / mass.sum())} for t in order[:6]],
            "effect_energy_hud_share": float(per_tile[HUD:].sum() / per_tile.sum())})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
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

    labels, keep = {}, {}
    for split in data:
        fatal = data[split]["p_death1"] > 0.5
        alive = ~fatal
        damaged = (dh[split][0] <= -2) & alive
        fatal_opp = fatal.any(1) & alive.any(1)
        damage_opp = damaged.any(1) & (~damaged & alive).any(1)
        keep[split] = torch.where(fatal_opp | damage_opp)[0]
        k = keep[split]
        labels[split] = {"fatal": (fatal[k], torch.ones_like(fatal[k])), "damage": (damaged[k], alive[k]),
                         "rows": {"fatal": torch.where(fatal_opp[k])[0], "damage": torch.where(damage_opp[k])[0]}}
    log(stage="roots", **{f"{s}_{o}": len(labels[s]["rows"][o]) for s in labels for o in ("fatal", "damage")})
    kept = {s: succ[s][0][keep[s]] for s in data}
    del data, succ, dh          # the contexts and unkept successors are ~6 GB; the first run was OOM-killed

    from d4mj.experiments import _load_bridge_parent
    bundle, heads, _ = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    del heads
    old = old_encoder(device)
    pca_old = torch.load(OLD / "evidence/state_cache.pt", weights_only=False, map_location="cpu")["pca"]
    feats = {s: encode(bundle, old, pca_old, kept[s], device) for s in kept}
    del bundle, old, kept
    torch.cuda.empty_cache()
    log(stage="encoded")

    pca = fit_pca(feats["fit"]["grid"].flatten(0, 1))
    for s in feats:
        feats[s]["grid_pca"] = apply_pca(pca, feats[s].pop("grid"))
    sample = feats["fit"]["tokens"].flatten(0, 2)
    sample = sample[torch.randperm(len(sample), generator=torch.Generator().manual_seed(args.seed))[:400_000]]
    centroids = kmeans(nn.functional.layer_norm(sample.float(), (WIDTH,)), K, 30, args.seed, device)
    quant = {}
    for s in feats:
        codes = []
        for j in range(0, len(feats[s]["tokens"]), 64):
            t = nn.functional.layer_norm(feats[s]["tokens"][j:j + 64].to(device).float(), (WIDTH,))
            codes.append(torch.cdist(t.flatten(0, 2), centroids).argmin(1).reshape(t.shape[:3]).cpu())
        feats[s]["codes"] = torch.cat(codes)
        t = nn.functional.layer_norm(feats[s]["tokens"][:256].to(device).float(), (WIDTH,))
        quant[s] = float(((t - centroids[feats[s]["codes"][:256].to(device)]) ** 2).sum() / (t ** 2).sum())
    used = int(torch.unique(feats["fit"]["codes"]).numel())
    log(stage="quantized", codes_used=used, relative_residual=quant)

    getters = {
        "z": (lambda s, i: feats[s]["z"][keep_rows[s][i]], None),
        "grid_pca": (lambda s, i: feats[s]["grid_pca"][keep_rows[s][i]], None),
        "u_old": (lambda s, i: feats[s]["u_old"][keep_rows[s][i]], None),
        "tokens": (lambda s, i: feats[s]["tokens"][keep_rows[s][i]].flatten(2), WIDTH),
        "tokens_ln": (lambda s, i: nn.functional.layer_norm(
            feats[s]["tokens"][keep_rows[s][i]].float(), (WIDTH,)).flatten(2), WIDTH),
        "codes": (lambda s, i: nn.functional.one_hot(feats[s]["codes"][keep_rows[s][i]], K).float().flatten(2), K),
    }
    results = {}
    for outcome in ("fatal", "damage"):
        keep_rows = {s: labels[s]["rows"][outcome] for s in labels}
        lab = {s: tuple(t[keep_rows[s]] for t in labels[s][outcome]) for s in labels}
        rows = {s: torch.arange(len(keep_rows[s])) for s in labels}
        results[outcome] = {}
        for name, (get, width) in getters.items():
            results[outcome][name] = measure(get, rows, lab, device, width)
            log(outcome=outcome, geometry=name, auc=round(results[outcome][name]["judge_within_root_auc"], 4),
                share=f'{results[outcome][name]["share"]:.3e}')

    r = results["fatal"]
    valid = lambda g: r[g]["judge_within_root_auc"] >= 0.90
    pooled = [g for g in ("z", "grid_pca") if valid(g)]
    tiles = [g for g in ("tokens_ln", "codes") if valid(g)]
    if not tiles or not pooled:
        reading = "void"
    else:
        base = max(r[g]["share"] for g in pooled)
        if any(r[g]["share"] >= 10 * base and r[g]["locality_tiles_for_half"] <= 3 for g in tiles):
            reading = "per_tile_premise_holds"
        elif all(r[g]["share"] < 3 * base or r[g]["locality_tiles_for_half"] > 10 for g in tiles):
            reading = "per_tile_premise_fails"
        else:
            reading = "mixed"
    evidence = {"schema": "d4mj_geometry_premise_v1", "status": "EXPLORATORY: observability roots",
                "script_sha256": _sha256(Path(__file__)), "identity": recorded,
                "checkpoint_sha256": _sha256(checkpoint), "codebook": {"K": K, "used": used,
                                                                       "relative_residual": quant},
                "reading": reading, "result": results}
    (args.out / "geometry.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="geometry_complete", reading=reading)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
