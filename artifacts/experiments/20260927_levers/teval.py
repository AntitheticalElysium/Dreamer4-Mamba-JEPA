"""Shared evaluation of per-tile worlds on the diagnostic futures (1,002 roots, seeds 900,000+, never trained on).

Token caches (layer-normed patch tokens, fp16), per encoder: context frames [R,4], the factual 16-step future
(sample 0) [R,16], the 17 one-step successors under key 0 [R,17].
Fact probes, fitted per token space on TRUE or GENERATED tokens of the 70% train seeds (the diagnosis split:
randperm(seed 0)), read on the 30% test seeds: tile class per map cell (shared linear 192->17 on the cell's own token), zombie at a
cell (192->1), HUD health/food/drink/energy (the 18 HUD tokens -> 4), facing (player token -> 4). Closed-form ridge,
lambda chosen on a held-out fifth of the train seeds.
Per world:
  onestep   all 17 actions from the 4 context frames: squared error / copying the last frame, by transition
            class (diagnosis onestep.classify, lava entries counted as moves); imagined change energy
            moved / blocked; tile accuracy of the 4 cells around the player on the imagined successor
  rollout   the factual 16 steps, window of the last 5 frames (4 at the first step; W=6 in training):
            error / total variance by depth for imagined and teacher-forced; per-step gain
            rms(g_k - t_k) / rms(g_{k-1} - s_{k-1}); facts by depth for imagined, teacher-forced, copy-root, true
  snap      optional: every imagined token replaced by its nearest codebook vector after each step
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DIAG = ROOT / "artifacts/experiments/20260926_diagnosis"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(DIAG))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))

ENCODERS = {"raw": ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt",
            "tc": ROOT / "artifacts/lewm_m4_canonical/tc/joint/step-010000.pt",
            "ldad10": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam10/step-010000.pt",
            "ldad1": ROOT / "artifacts/eda/levers_ldad_v1/raw_lam1/step-010000.pt"}
CACHE = ROOT / "artifacts/eda/levers_futures_tokens_{}.pt"
META = ROOT / "artifacts/eda/diagnosis_rollouts_v1/meta.pt"
N, H = 17, 16
NEAR = (22, 40, 30, 32)                                            # up, down, left, right of token 31
MAP = [r * 9 + c for r in range(7) for c in range(9)]
EDGE = [r * 9 + c for r in range(7) for c in range(9) if r in (0, 6) or c in (0, 8)]
INTERIOR = [i for i in MAP if i not in EDGE and i not in NEAR and i != 31]


def split():
    meta = torch.load(META)
    seeds = meta["seed"]
    u = seeds.unique()
    train_seeds = u[torch.randperm(len(u), generator=torch.Generator().manual_seed(0))[: int(0.7 * len(u))]]
    return meta, torch.isin(seeds, train_seeds), train_seeds


@torch.no_grad()
def build_cache(name, device):
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    from rollouts import load_roots
    path = Path(str(CACHE).format(name))
    if path.exists():
        return torch.load(path, weights_only=False)
    raw_payload = torch.load(ENCODERS[name], map_location="cpu", weights_only=False)
    payload = read_lewm_bundle(ENCODERS[name]) if "format" in raw_payload else raw_payload     # LDAD runs: plain dict
    bundle = ModelBundle.create(config_from_dict(payload["config"]))
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    enc = bundle.encoder.to(device).freeze()
    roots = load_roots()

    def tok(frames):
        out = []
        for i in range(0, len(frames), 128):
            _, _, t, _, _ = enc._hidden(frames[i:i + 128, None].to(device))
            out.append(F.layer_norm(t.float(), (192,)).half().cpu())
        return torch.cat(out)
    R = len(roots)
    cache = {"ctx": tok(torch.stack([r["context"] for r in roots]).flatten(0, 1)).view(R, 4, 81, 192),
             "fut": tok(torch.stack([r["future_frames"][0] for r in roots]).flatten(0, 1)).view(R, H, 81, 192),
             "one": tok(torch.stack([r["onestep_frames"][0] for r in roots]).flatten(0, 1)).view(R, N, 81, 192),
             "ctx_a": torch.stack([r["context_actions"] for r in roots]),
             "fut_a": torch.stack([r["future_actions"] for r in roots])}
    torch.save(cache, path)
    return cache


def facts_of(vis):
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 7, 9, 17).argmax(-1).flatten(-2)      # [.., 63]
    zombie = (v[..., 1071:1512].reshape(*v.shape[:-1], 7, 9, 7)[..., 0] > 0).float().flatten(-2)
    return {"tile": tiles, "zombie": zombie, "hud": v[..., 1512:1516], "facing": v[..., 1516:1520].argmax(-1)}


def ridge(x, y, tr, va, lams=(1e-3, 1e-2, 1e-1, 1, 10)):
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = lambda q: torch.cat([((q - mu) / sd).double(), torch.ones(len(q), 1, dtype=torch.float64)], 1)
    X, Y = xs(x[tr]), y[tr].double()
    Xv, Yv = xs(x[va]), y[va].double()
    best = None
    for lam in lams:
        w = torch.linalg.solve(X.T @ X + lam * len(X) * torch.eye(X.shape[1], dtype=X.dtype), X.T @ Y)
        err = float(((Xv @ w - Yv) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, w)
    w = best[1]
    return lambda q: (xs(q) @ w).float()


class Probes:
    """Ridge readouts fitted on the supplied 17 frames from TRAIN-seed roots."""

    def __init__(self, cache, meta, train_roots, train_seeds, tokens=None, visible=None):
        seeds = meta["seed"]
        val_roots = torch.isin(seeds, train_seeds[: len(train_seeds) // 5])
        fit_roots = train_roots & ~val_roots
        toks = (tokens if tokens is not None else torch.cat([cache["ctx"][:, -1:], cache["fut"]], 1)).float()
        vis = visible if visible is not None else torch.cat(
            [meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1)
        assert toks.shape == (len(seeds), 17, 81, 192) and vis.shape == (len(seeds), 17, 1534)
        f = facts_of(vis)
        rows = lambda m: m[:, None].expand(-1, 17).flatten()
        fr, vr = rows(fit_roots), rows(val_roots)
        t = toks.flatten(0, 1)
        cell = t[:, MAP].flatten(0, 1)                                                     # [(R*17)*63,192]
        cr = lambda m: m[:, None].expand(-1, 63).flatten()
        sub = torch.randperm(len(cell), generator=torch.Generator().manual_seed(0))[:400_000]
        keep_f, keep_v = torch.zeros(len(cell), dtype=torch.bool), torch.zeros(len(cell), dtype=torch.bool)
        keep_f[sub] = cr(fr)[sub]; keep_v[sub] = cr(vr)[sub]
        self.tile = ridge(cell, F.one_hot(f["tile"].flatten(), 17).float(), keep_f, keep_v)
        self.zombie = ridge(cell, f["zombie"].flatten()[:, None], keep_f, keep_v)
        self.hud = ridge(t[:, 63:81].flatten(1), f["hud"].flatten(0, 1), fr, vr)
        self.facing = ridge(t[:, 31], F.one_hot(f["facing"].flatten(), 4).float(), fr, vr)

    def read(self, tokens, vis):
        """tokens [n,81,192] (any), vis [n,1534] -> fact metrics."""
        from compound import auc
        f = facts_of(vis)
        tok = tokens.float()
        pred_tile = self.tile(tok[:, MAP].flatten(0, 1)).view(len(tok), 63, 17).argmax(-1)
        correct = (pred_tile == f["tile"]).float()
        idx = {i: MAP.index(i) for i in MAP}
        region = lambda cells: float(correct[:, [idx[c] for c in cells]].mean())
        z = self.zombie(tok[:, MAP].flatten(0, 1)).view(len(tok), 63)
        near = [idx[c] for c in NEAR]
        hud = self.hud(tok[:, 63:81].flatten(1))
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9))
        return {"tile_near": region(NEAR), "tile_interior": region(INTERIOR), "tile_edge": region(EDGE),
                "zombie_auc": auc(z, f["zombie"]), "zombie_near_auc": auc(z[:, near], f["zombie"][:, near]),
                "health_r2": r2(hud[:, 0], f["hud"][:, 0]), "food_r2": r2(hud[:, 1], f["hud"][:, 1]),
                "facing_acc": float((self.facing(tok[:, 31]).argmax(-1) == f["facing"]).float().mean())}


def snap(x, codes):
    from tworld import quantize
    return codes[quantize(x.to(codes.device), codes)].to(x.dtype).cpu() if codes is not None else x


@torch.no_grad()
def step(world, frames, actions, device, config):
    """frames [B,T,81,192], actions [B,T] (last = the action to take) -> next frame [B,81,192] (LN)."""
    from d4mj.train import autocast_context
    from tworld import quantize
    frames = frames.to(device).float()
    if world.head == "categorical":                      # trained on quantized inputs
        frames = world.codes[quantize(frames, world.codes)]
    with autocast_context(config):
        out = world(frames, actions.to(device))[0]
    return out[:, -1].float().cpu()


@torch.no_grad()
def evaluate(world, cache, meta, probes, train_roots, train_seeds, test_roots, device,
             codes=None, batch=16, window=5):
    from d4mj.config import config_from_dict
    import spatial as S
    from onestep import CLASSES, classify
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    R = len(meta["seed"])
    cls, _ = classify(meta)
    one_pred = torch.empty(R, N, 81, 192, dtype=torch.float16)
    gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
    tf = torch.empty(R, H, 81, 192, dtype=torch.float16)
    for i in range(0, R, batch):
        ctx = cache["ctx"][i:i + batch].float()
        b = len(ctx)
        ca, fa, fut = cache["ctx_a"][i:i + batch], cache["fut_a"][i:i + batch], cache["fut"][i:i + batch].float()
        fan = ctx.repeat_interleave(N, 0)
        acts = torch.cat([ca.repeat_interleave(N, 0), torch.arange(N).repeat(b)[:, None]], 1)
        one_pred[i:i + b] = snap(step(world, fan, acts, device, config), codes).view(b, N, 81, 192).half()
        g_frames, t_frames, a_hist = [ctx[:, j] for j in range(4)], [ctx[:, j] for j in range(4)], [ca[:, j] for j in range(3)]
        for k in range(H):
            w = 4 if k == 0 else window
            a = torch.stack(a_hist[-(w - 1):] + [fa[:, k]], 1)
            g = snap(step(world, torch.stack(g_frames[-w:], 1), a, device, config), codes)
            t = step(world, torch.stack(t_frames[-w:], 1), a, device, config)
            gen[i:i + b, k], tf[i:i + b, k] = g.half(), t.half()
            g_frames.append(g); t_frames.append(fut[:, k]); a_hist.append(fa[:, k])
    generated_probe = Probes(cache, meta, train_roots, train_seeds,
                             tokens=torch.cat([cache["ctx"][:, -1:], gen], 1))
    one_step_probe = Probes(cache, meta, train_roots, train_seeds,
                            tokens=one_pred, visible=meta["onestep_visible"][:, 0])
    true_one, root = cache["one"].float(), cache["ctx"][:, -1].float()
    err = ((one_pred.float() - true_one) ** 2).sum((-1, -2))
    cp = ((root[:, None] - true_one) ** 2).sum((-1, -2))
    res = {"onestep": {c: float(err[cls == i].mean() / cp[cls == i].mean()) for i, c in enumerate(CLASSES)}}
    res["onestep"]["all"] = float(err.mean() / cp.mean())
    ch = ((one_pred.float() - root[:, None]) ** 2).sum((-1, -2))
    tch = ((true_one - root[:, None]) ** 2).sum((-1, -2))
    res["onestep"]["imagined_moved_over_blocked"] = float(ch[cls == 0].mean() / ch[cls == 1].mean())
    res["onestep"]["true_moved_over_blocked"] = float(tch[cls == 0].mean() / tch[cls == 1].mean())
    moved = (cls == 0) & test_roots[:, None]
    ov = meta["onestep_visible"][:, 0]
    res["onestep"]["moved_facts_imagined_true_fit"] = probes.read(one_pred.float()[moved], ov[moved])
    res["onestep"]["moved_facts_imagined_generated_fit"] = one_step_probe.read(
        one_pred.float()[moved], ov[moved])
    res["onestep"]["moved_facts_true"] = probes.read(true_one[moved], ov[moved])
    res["onestep"]["moved_facts_copy"] = probes.read(root[:, None].expand_as(true_one)[moved], ov[moved])
    fut = cache["fut"].float()
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    V = float(((fut - fut.flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    depth = []
    for k in range(H):
        m = alive[:, k]
        e = lambda x: float(((x[m].float() - fut[m, k]) ** 2).sum((-1, -2)).mean() / V)
        row = {"k": k + 1, "gen": e(gen[:, k]), "tf": e(tf[:, k]), "copy_root": e(root)}
        if k:
            inh = ((gen[m, k - 1].float() - fut[m, k - 1]) ** 2).sum((-1, -2)).mean()
            row["gain"] = float((((gen[m, k].float() - tf[m, k].float()) ** 2).sum((-1, -2)).mean() / inh) ** 0.5)
        depth.append(row)
    res["rollout"] = depth
    generr = ((gen.float() - fut) ** 2).sum((-1, -2))                                # [R,16]
    tferr = ((tf.float() - fut) ** 2).sum((-1, -2))
    res["_per_root"] = {"onestep_err": err, "onestep_copy": cp, "class": cls, "gen_err": generr, "tf_err": tferr,
                        "alive": alive, "V": V, "seed": meta["seed"]}
    fv = meta["future_visible"][:, 0]
    res["schema_version"] = 2
    res["readout_contract"] = {
        "true_fit": "ridge fitted on real TRAIN-seed root and factual successor tokens",
        "generated_fit": "same ridge fitted on true TRAIN-seed root tokens and this world's generated factual successors",
        "one_step_generated_fit": "same ridge fitted on this world's all-17-action generated TRAIN-seed successors",
        "validation": "first fifth of shuffled TRAIN seeds; judgment on disjoint TEST seeds",
    }
    res["facts_true_fit"] = {}
    res["facts_generated_fit"] = {}
    for k in (1, 2, 4, 8, 16):
        m = alive[:, k - 1] & test_roots
        res["facts_true_fit"][k] = {name: probes.read(x[m].float(), fv[m, k - 1]) for name, x in
                           (("imagined", gen[:, k - 1]), ("teacher", tf[:, k - 1]), ("copy_root", root),
                            ("true", fut[:, k - 1]))}
        res["facts_generated_fit"][k] = {"imagined": generated_probe.read(
            gen[m, k - 1].float(), fv[m, k - 1])}
    return res


def load_world(path, device):
    from tworld import TWorld
    st = torch.load(path, map_location="cpu", weights_only=False)
    head = st["args"]["head"]
    codebook = None
    if head == "categorical":
        codebook = st["world"]["codes"]
    levels = st["world"]["noise_embed.weight"].shape[0] if "noise_embed.weight" in st["world"] else 0
    w = TWorld(head, codebook, st["args"].get("backbone", "full"), st["args"].get("regions", "all"), levels,
               st["args"].get("skip") == "True").to(device)
    w.load_state_dict(st["world"])
    return w.eval(), st


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("worlds", nargs="+", type=Path)
    parser.add_argument("--snap", type=Path, default=None, help="codebook to snap imagined tokens to")
    parser.add_argument("--hard", action="store_true", help="corr heads: argmax decoding (ITC's binarized plan)")
    parser.add_argument("--window", type=int, default=5, help="rollout window in frames (default 5)")
    args = parser.parse_args(argv)
    device = torch.device("cuda")
    meta, train_roots, train_seeds = split()
    test_roots = ~train_roots
    out = HERE / "evals"
    out.mkdir(exist_ok=True)
    probes_by = {}
    for path in args.worlds:
        world, st = load_world(path, device)
        world.hard_decode = args.hard
        pool = st["args"]["pool"]
        cache = build_cache(pool, device)
        if pool not in probes_by:
            probes_by[pool] = Probes(cache, meta, train_roots, train_seeds)
        codes = torch.load(args.snap, weights_only=False)["codes"].float().to(device) if args.snap else None
        per_token_ssm = getattr(world, "backbone_kind", "full") in ("fmamba", "fcanvas")    # B*82 x 16k-float states
        res = evaluate(world, cache, meta, probes_by[pool], train_roots, train_seeds, test_roots, device, codes,
                       batch=4 if per_token_ssm else 16,
                       window=args.window)
        tag = st["name"] + (f"__snap_{args.snap.stem}" if args.snap else "") + ("__hard" if args.hard else "") \
            + ("" if args.window == 5 else f"__w{args.window}")
        per_root = out / f"{tag}_per_root.pt"
        if per_root.exists():
            per_root = out / f"{tag}__readout_v2_per_root.pt"
        torch.save(res.pop("_per_root"), per_root)
        report = json.dumps(res, indent=2) + "\n"
        (out / f"{tag}__readout_v2.json").write_text(report)
        legacy = out / f"{tag}.json"
        if not legacy.exists():
            legacy.write_text(report)
        brief = {"onestep": {k: round(v, 3) for k, v in res["onestep"].items() if isinstance(v, float)},
                 "gen": [round(res["rollout"][d]["gen"], 3) for d in (0, 3, 7, 15)],
                 "tf": [round(res["rollout"][d]["tf"], 3) for d in (0, 3, 7, 15)],
                 "facts16_true_fit": {k: round(v, 3) for k, v in res["facts_true_fit"][16]["imagined"].items()},
                 "facts16_generated_fit": {k: round(v, 3) for k, v in res["facts_generated_fit"][16]["imagined"].items()}}
        print(tag, json.dumps(brief), flush=True)


if __name__ == "__main__":
    main()
