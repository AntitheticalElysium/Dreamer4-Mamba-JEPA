"""Learned-fixation design reading G (PLAN.md G): where does the learning a look produces land, over P1? Per world,
checkpoint 1-10 and F's rows, the pixel error at every one of the 49 windows of the predictions a look at that window
trains (TSP from the real periphery; the forward glimpse from history; the reconstruction), then their progress
between consecutive checkpoints on the same held-out rows. Labels select rows and label windows only.
Usage (project venv, GPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.landscape OUT_DIR [--smoke]
                           python -m artifacts.experiments.20261031_learned_fixation_sweep.landscape --summary OUT_DIR
"""
import importlib, json, math, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

F = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.locus")
PR = importlib.import_module("artifacts.experiments.20261029_fovea_probes.analyse")
DEV = F.DEV
WORLDS = {"w1": TR.ROOT / "artifacts/saccade_m0/p1_world1", "p1f": TR.ROOT / "artifacts/experiments/20261030_p1_guided/run"}
INSTR = ("tsp", "view", "rec")
NB, NW = 8, 49                                                           # rows per batch, windows per row
RULE = ("drink", "wood", "sapling")                                      # P1-F's rule fixated these attempts


def frames_of(L, rows):
    """Real frame t of every row, uint8 [n, 63, 63, 3], from the corpus."""
    eps, out = S.episodes(L), [None] * len(rows)
    need = {}
    for i, (e, t) in enumerate(rows):
        need.setdefault(e, []).append((i, t))
    for e, its in need.items():
        obs = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy()
        for i, t in its:
            out[i] = obs[t]
    return torch.as_tensor(np.stack(out))


@torch.no_grad()
def run(model, cfg, enc, rows, frames, gen):
    """Per row and window: pixel MSE of TSP, the forward glimpse and the reconstruction [n, 49]; and TSP's NLL at the
    row's recorded fixation against S.encode's (the alignment check)."""
    den, s, sd = model.denoiser, cfg["edm"]["sampler"], cfg["edm"]["sigma_data"]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    hat = s["sigma_max"] * (1 + min(s["s_churn"] / s["steps"], math.sqrt(2) - 1))    # production's first evaluation
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    cen = F.CENTRES.to(DEV)
    out = {k: [] for k in (*INSTR, "check")}
    for b0 in range(0, len(rows), NB):
        br = rows[b0:b0 + NB]
        n = len(br)
        take = lambda k, dt: torch.stack([enc[e][k][t + dt] for e, t in br]).to(DEV)
        fix, img = cen.repeat(n, 1), frames[b0:b0 + n].to(DEV).repeat_interleave(NW, 0)
        cells, bp = M.window(fix), take("bp", 0).float().repeat_interleave(NW, 0)
        pix = M.cell_pixels(M.crop(img, fix))
        y = pix.float() / 127.5 - 1
        with TR.autocast(DEV, cfg):
            Pc = model.perception(img, fix)
            mt, st = model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), Pc.v, cells)
            mr, _ = model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), Pc.v, cells)
            cache = F.context_cache(model, cfg, take, 1, gen)              # the context does not depend on the query:
            cache = {"t": cache["t"], "kv": [(k.repeat_interleave(NW, 0), v.repeat_interleave(NW, 0))   # once per row,
                                            for k, v in cache["kv"]]}                                   # same draws
            x = tuple(hat * torch.randn((n, 1, *shp), device=DEV, generator=gen).repeat_interleave(NW, 0) for shp in shapes)
            Dd, _ = M.denoised(den, x, torch.full((n * NW, 1), hat, device=DEV),
                               take("prev_action", 0).long().repeat_interleave(NW, 0)[:, None],
                               model.belief.memory(bp)[:, None], fix[:, None], cache, clamp, sd)
            gv, vv = Dd[1][:, 0].float(), Dd[2][:, 0].float()
            mv, _ = model.pixels(model.tsp.reconstruct(gv, vv, cells), vv, cells)
            nll = M.pixel_nll(mt, st, pix).float().mean((-1, -2)).view(n, NW)
        mse = lambda m: (m.float().clamp(-1, 1) - y).square().flatten(1).mean(1).view(n, NW)
        for k, m in zip(INSTR, (mt, mv, mr)):
            out[k].append(mse(m).cpu())
        rec = take("fix", 0).long()
        idx = (rec[:, 0] - 1) * 7 + rec[:, 1] - 1
        out["check"].append(torch.stack([nll[torch.arange(n), idx].cpu(), torch.stack([enc[e]["tsp"][t] for e, t in br])], 1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def window_labels(L, rows, ctx, d):
    """Per row: class, episode, light; per (row, window): covers the consequence (F's: the class's counter, the faced
    tile for mining), the faced tile, an entity (zombie / cow / skeleton / arrow, state t), the HUD."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    cls = np.array([c for c in ctx for _ in ctx[c]])
    g0 = np.array([s_off[e] + t - 1 for e, t in rows])                   # the state DO acted from (F's)
    g1 = g0 + 1                                                           # frame t's state
    win = F.WIN.numpy()                                                   # [49, 9]
    faced = (3 + d[g0, 0]) * 9 + 4 + d[g0, 1]
    counter = np.array([F.ATTEMPTS[c][1] if c in F.ATTEMPTS else F.ATTEMPTS["stone"][1] if c == "stone_no_tool" else -1
                        for c in cls])
    covers = lambda cell: (win[None] == cell[:, None, None]).any(-1)       # [n, 49]
    cons = covers(counter) | (covers(faced) & np.isin(cls, F.MINED)[:, None])
    scr = win[None].repeat(len(rows), 0)
    pos = L["s_pos"][g1].astype(int)
    wk = np.stack([pos[:, None, None, 0] + scr // 9 - 3, pos[:, None, None, 1] + scr % 9 - 4], -1)
    wk = np.where((scr < 63)[..., None], wk, -1000)
    kinds = np.stack([PR.content(L, g1, wk[:, :, j]) for j in range(9)], -1)
    return {"cls": cls, "ep": np.array([e for e, _ in rows]), "t": np.array([t for _, t in rows]),
            "light": L["s_light"][g1], "consequence": cons, "faced": covers(faced),
            "entity": ((kinds >= 1) & (kinds <= 4)).any(-1), "hud": np.broadcast_to((win >= 63).any(-1), cons.shape).copy()}


def spearman(a, b):
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def summary(out_dir):
    """PLAN G's readings from the saved arrays (CPU)."""
    lab, res = np.load(out_dir / "labels.npz"), {}
    meta = json.loads((out_dir / "meta.json").read_text())
    ep, cls, light = lab["ep"], lab["cls"], lab["light"]
    est = lambda v, m: G.estimate(np.asarray(v, float), m, ep)
    for world in WORLDS:
        files = sorted(out_dir.glob(f"{world}_ckpt_*.npz"))
        if len(files) < 2:
            continue
        E = {k: np.stack([np.load(f)[k] for f in files]) for k in INSTR}          # [ckpts, rows, 49]
        ks = [int(f.stem.split("_")[-1]) for f in files]
        res[world] = {}
        for c in np.unique(cls):
            m = cls == c
            groups = {"all": m}
            if c == "random":
                groups = {"all": m, "bright": m & (light >= 0.7), "dark": m & (light <= 0.3)}
            for gname, gm in groups.items():
                key = c if gname == "all" else f"{c}_{gname}"
                r = res[world][key] = {"rows": int(gm.sum()), "intervals": {}}
                odd = (ep % 2).astype(bool)
                for i in range(len(ks) - 1):
                    iv = r["intervals"][f"{ks[i]}-{ks[i + 1]}"] = {}
                    for k in INSTR:
                        lp, err = E[k][i] - E[k][i + 1], E[k][i]
                        v = lp[gm].mean(0)
                        h1, h2 = lp[gm & odd].mean(0), lp[gm & ~odd].mean(0)
                        e1, e2 = err[gm & odd].mean(0), err[gm & ~odd].mean(0)
                        pos = np.clip(v, 0, None)
                        x = {"mean_lp": float(v.mean()), "split_half_rho": spearman(h1, h2),
                             "top5_share_of_positive": float(np.sort(pos)[-5:].sum() / max(pos.sum(), 1e-12)),
                             "rho_lp_error": spearman(v, err[gm].mean(0)), "error_split_half_rho": spearman(e1, e2)}
                        rowmean = lp.mean(1)
                        for t in ("consequence", "faced", "entity", "hud"):
                            w = lab[t]
                            have = gm & w.any(1)
                            if have.sum() >= 20:
                                tm = (lp * w).sum(1) / np.maximum(w.sum(1), 1)
                                x[f"{t}_excess"] = est(tm - rowmean, have)            # type's windows minus the row's mean
                                x[f"{t}_error_excess"] = est((err * w).sum(1) / np.maximum(w.sum(1), 1) - err.mean(1), have)
                        if c not in ("random",) and gm.any():
                            am = lp.argmax(1)
                            x["argmax_covers_consequence"] = est(lab["consequence"][np.arange(len(am)), am], gm)
                            x["error_argmax_covers_consequence"] = est(lab["consequence"][np.arange(len(am)), err.argmax(1)], gm)
                            x["chance"] = float(lab["consequence"][gm].mean())
                            best, med = int(v.argmax()), int(np.argsort(v)[NW // 2])
                            dlt = lp[gm][:, best] - lp[gm][:, med]
                            need = (2 * dlt.std() / max(dlt.mean(), 1e-12)) ** 2 if dlt.mean() > 0 else None
                            freq = meta["class_frequency"].get(c)
                            x["bandit"] = {"best_window": best, "looks_needed_2se": need,
                                           "looks_available_floor_0.1": None if freq is None else freq * 25.6e6 * 0.1 / NW}
                        iv[k] = x
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    for world, r in res.items():
        print("==", world)
        for key, x in r.items():
            line = []
            for iv, y in x["intervals"].items():
                z = y["view"]
                cx = z.get("consequence_excess")
                line.append(f"{iv}: rho {z['split_half_rho']:+.2f} lp~err {z['rho_lp_error']:+.2f}" +
                            (f" cons {cx['est']:+.4f}({cx['se']:.4f})" if cx else ""))
            print(f"  {key:16s} n{x['rows']:4d} view |", " | ".join(line))
    return res


def main():
    if sys.argv[1] == "--summary":
        summary(Path(sys.argv[2]))
        return
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12 if smoke else F.N_EP)
    cap = (F.CAP, F.N_RANDOM)
    F.CAP = F.N_RANDOM = 10 ** 9                                          # uncapped class counts (bandit reading)
    full, _, _ = F.contexts(L, np.random.default_rng(0))
    F.CAP, F.N_RANDOM = cap
    ctx, face, d = F.contexts(L, np.random.default_rng(2031))             # F's rows exactly
    if smoke:
        ctx = {c: rows[:6] for c, rows in ctx.items() if rows}
    rows = [r for c in ctx for r in ctx[c]]
    frames = frames_of(L, rows)
    frames_total = int((L["ep_T"] + 1).sum())
    meta = {"episodes": int(len(L["ep_T"])), "contexts": {c: len(v) for c, v in ctx.items()},
            "class_frequency": {c: len(v) / frames_total for c, v in full.items()}, "checkpoints": {}}
    np.savez_compressed(out_dir / "labels.npz", **window_labels(L, rows, ctx, d))
    for world, wdir in WORLDS.items():
        for k in ((1, 2) if smoke else range(1, 11)):
            path = wdir / f"ckpt_{k:02d}.pt"
            name = f"{world}_ckpt_{k:02d}"
            ck = torch.load(path, weights_only=False, map_location="cpu")
            cfg = ck["cfg"]
            model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
            del ck
            enc = S.encode(model, L, cfg)
            gen = torch.Generator(device=DEV).manual_seed(2032)              # the same draws at every checkpoint
            o = run(model, cfg, enc, rows, frames, gen)
            chk = o.pop("check")
            dev = np.abs(chk[:, 0] - chk[:, 1]) / np.abs(chk[:, 1]).mean()
            rel = float(np.median(dev))                                  # alignment shifts the median; S.encode's half-
            meta["checkpoints"][name] = {"path": str(path), "sha256": TR.file_sha256(path),   # precision bp alone moves
                                         "tsp_nll_check_median_rel": rel, "max_rel": float(dev.max())}   # a row <= 2.5%
            print(name, {k: round(float(v.mean()), 5) for k, v in o.items()}, "tsp check median", round(rel, 5),
                  "max", round(float(dev.max()), 5), flush=True)
            if rel > 0.02:
                raise RuntimeError(f"TSP at the recorded fixation differs from S.encode: {rel}")
            np.savez_compressed(out_dir / f"{name}.npz", **{k: v.astype(np.float32) for k, v in o.items()})
            (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
            del model, enc
            torch.cuda.empty_cache()
        if smoke:
            break
    summary(out_dir)


if __name__ == "__main__":
    main()
