"""E4b. The common-currency test for LDAD runs: do the decision facts survive IMAGINATION, not just the encoder?

"x copy" and "/V" live in each run's own latent geometry, and LDAD reshapes z_{t+1} - z_t to encode the action the
world is given, so they are not comparable across runs. Facts are. Per run (joint world, le-wm's 3-frame window
throughout, eval-mode z), on the diagnostic futures (1,002 roots; diagnosis 70/30 seed split):
  probes    identical ridge probes fitted on true root z plus TRUE or GENERATED factual successor z
            from train-seed roots (17 frames): zombie adjacent, zombie in view, each neighbour tile passable (4), health,
            food, drink, energy; AUC / R^2
  read on   true z at depth k (readout positive control), copy-root z, imagined z at depth k,
            k in 1, 4, 8, 16, test-seed roots alive at k
  onestep   all-17-action fan: moved vs blocked and health-drop AUC with separate true-fit and generated-fit
            probes, evaluated on the same imagined z(a)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import ldad_eval as E  # noqa: E402
from compound import auc  # noqa: E402

H = 16
NB = ((2, 4), (4, 4), (3, 3), (3, 5))                    # up, down, left, right of the player (visible grid)
SOLID = {1, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16}


def facts(vis):
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 7, 9, 17).argmax(-1)
    mobs = v[..., 1071:1512].reshape(*v.shape[:-1], 7, 9, 7)
    zom = mobs[..., 0] > 0
    occ = mobs[..., :3].sum(-1) > 0
    out = {"zombie_adjacent": torch.stack([zom[..., r, c] for r, c in NB], -1).any(-1).float(),
           "zombie_in_view": zom.flatten(-2).any(-1).float()}
    for (r, c), name in zip(NB, ("up", "down", "left", "right")):
        t = tiles[..., r, c]
        solid = torch.zeros_like(t, dtype=torch.bool)
        for s in SOLID:
            solid |= t == s
        out[f"pass_{name}"] = (~solid & ~occ[..., r, c]).float()
    for i, name in enumerate(("health", "food", "drink", "energy")):
        out[name] = v[..., 1512 + i]
    return out


BINARY = ("zombie_adjacent", "zombie_in_view", "pass_up", "pass_down", "pass_left", "pass_right")


def ridge(x, y, fit, val, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    mu, sd = x[fit].mean(0), x[fit].std(0).clamp_min(1e-6)
    xs = lambda q: torch.cat([((q - mu) / sd).double(), torch.ones(len(q), 1, dtype=torch.float64)], 1)
    best = None
    for lam in lams:
        X = xs(x[fit])
        w = torch.linalg.solve(X.T @ X + lam * len(X) * torch.eye(X.shape[1], dtype=X.dtype), X.T @ y[fit].double())
        e = float(((xs(x[val]) @ w - y[val].double()) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, w)
    return lambda q: (xs(q) @ best[1]).float()


def score(pred, y, name):
    if name in BINARY:
        return auc(pred, y) if 0 < float(y.mean()) < 1 else float("nan")
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9))


@torch.no_grad()
def run_one(bundle, device):
    from d4mj.train import autocast_context
    from rollouts import load_roots
    from teval import split
    meta, train_roots, train_seeds = split()
    val_roots = torch.isin(meta["seed"], train_seeds[: len(train_seeds) // 5])
    fit_roots = train_roots & ~val_roots
    test = ~train_roots
    roots = load_roots()
    enc, world, config = bundle.encoder, bundle.world, bundle.config
    zf = lambda f: torch.cat([enc(f[i:i + 64][:, None].to(device)).float()[:, 0, 0].cpu() for i in range(0, len(f), 64)])
    R = len(roots)
    ctx = zf(torch.stack([r["context"] for r in roots]).flatten(0, 1)).view(R, 4, -1)
    true = zf(torch.stack([r["future_frames"][0] for r in roots]).flatten(0, 1)).view(R, H, -1)
    one = zf(torch.stack([r["onestep_frames"][0] for r in roots]).flatten(0, 1)).view(R, 17, -1)
    ca = torch.stack([r["context_actions"] for r in roots]); fa = torch.stack([r["future_actions"] for r in roots])
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    gen, fan = [], []
    with autocast_context(config):
        for i in range(0, R, 32):
            lat = [ctx[i:i + 32, j].to(device) for j in (1, 2, 3)]
            acts = [ca[i:i + 32, 1].to(device), ca[i:i + 32, 2].to(device)]
            st = world.teacher(torch.stack(lat, 1)[:, :, None], torch.stack(acts, 1)).state
            n = len(lat[0])
            adv, _ = world.advance(bundle.repeat_state(st, 17) if hasattr(bundle, "repeat_state") else st,
                                   torch.arange(17, device=device).repeat(n)[:, None])
            fan.append(adv.latent[:, 0, 0].float().cpu().view(n, 17, -1))
            g = []
            for k in range(H):
                a = fa[i:i + 32, k].to(device)
                s = world.teacher(torch.stack(lat[-3:], 1)[:, :, None], torch.stack(acts[-2:], 1)).state
                nx, _ = world.advance(s, a[:, None])
                q = nx.latent[:, 0, 0].float()
                g.append(q.cpu()); lat.append(q.to(lat[0].dtype)); acts.append(a)
            gen.append(torch.stack(g, 1))
    gen, fan = torch.cat(gen), torch.cat(fan)
    vis = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1)            # [R,17,...]
    F = facts(vis)
    X = torch.cat([ctx[:, -1:], true], 1)                                                        # [R,17,D]
    Xg = torch.cat([ctx[:, -1:], gen], 1)
    rows = lambda m: m[:, None].expand(-1, 17).flatten()
    out = {"schema_version": 2, "depth": {},
           "readout_contract": "same ridge capacity, TRAIN/validation split and TEST labels; true root plus true or generated factual successors"}
    probes = {name: ridge(X.flatten(0, 1), y.flatten(), rows(fit_roots), rows(val_roots)) for name, y in F.items()}
    generated_probes = {name: ridge(Xg.flatten(0, 1), y.flatten(), rows(fit_roots), rows(val_roots))
                        for name, y in F.items()}
    for k in (1, 4, 8, 16):
        m = test & alive[:, k - 1]
        row = {}
        for name, p in probes.items():
            y = F[name][m, k]
            row[name] = {"true": score(p(true[m, k - 1]), y, name),
                         "copy_root": score(p(ctx[m, -1]), y, name),
                         "imagined_true_fit": score(p(gen[m, k - 1]), y, name),
                         "imagined_generated_fit": score(generated_probes[name](gen[m, k - 1]), y, name)}
        out["depth"][k] = row
    # One-step fan: same-capacity true-fit and generated-fit probes for moved/blocked on imagined z(a).
    from onestep import classify
    cls, events = classify(meta)
    mv = torch.zeros_like(cls, dtype=torch.bool); mv[:, 1:5] = True
    sel = mv & (cls <= 1)
    lab = (cls == 0)
    tr = train_roots[:, None].expand_as(cls)
    p = ridge(one[sel], lab[sel].float(), (tr & ~val_roots[:, None])[sel], val_roots[:, None].expand_as(cls)[sel])
    pg = ridge(fan[sel], lab[sel].float(), (tr & ~val_roots[:, None])[sel], val_roots[:, None].expand_as(cls)[sel])
    te = (~tr)[sel]
    out["moved_vs_blocked_auc"] = {
        "true": auc(p(one[sel])[te], lab[sel][te]),
        "imagined_true_fit": auc(p(fan[sel])[te], lab[sel][te]),
        "imagined_generated_fit": auc(pg(fan[sel])[te], lab[sel][te])}
    hd = events["health_down"][0]
    ph = ridge(one.flatten(0, 1), hd.flatten().float(), (tr & ~val_roots[:, None]).flatten(), val_roots[:, None].expand_as(cls).flatten())
    phg = ridge(fan.flatten(0, 1), hd.flatten().float(),
                (tr & ~val_roots[:, None]).flatten(), val_roots[:, None].expand_as(cls).flatten())
    tef = (~tr).flatten()
    out["health_down_auc"] = {"true": auc(ph(one.flatten(0, 1))[tef], hd.flatten()[tef]),
                              "imagined_true_fit": auc(ph(fan.flatten(0, 1))[tef], hd.flatten()[tef]),
                              "imagined_generated_fit": auc(phg(fan.flatten(0, 1))[tef], hd.flatten()[tef])}
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=None, help="checkpoint names to rescore")
    args = parser.parse_args()
    device = torch.device("cuda")
    runs = dict(E.SCREENS) | dict(E.RUNS)
    if args.runs:
        assert set(args.runs) <= set(runs), sorted(set(args.runs) - set(runs))
        runs = {name: runs[name] for name in args.runs}
    path = HERE / "ldad_facts_v2.json"
    result = json.loads(path.read_text()) if path.exists() else {}
    for run, ckpt in runs.items():
        if run in result or not ckpt.exists():
            continue
        bundle = E.load(ckpt, device)
        result[run] = run_one(bundle, device)
        d = result[run]["depth"]
        print(run, json.dumps({k: {n: round(d[k][n]["imagined_generated_fit"], 3) for n in ("zombie_adjacent", "pass_left", "health")}
                               for k in d}), json.dumps(result[run]["moved_vs_blocked_auc"]), flush=True)
        current = json.loads(path.read_text()) if path.exists() else {}      # another lane may have written since
        path.write_text(json.dumps(current | {run: result[run]}, indent=2) + "\n")
        del bundle
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
