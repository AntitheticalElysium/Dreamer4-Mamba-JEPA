"""Learned-fixation design reading H (PLAN.md H): does choosing WHICH crop of each frame is trained on make the
look-local predictions learn faster than uniform crops? A controlled data-selection test on frozen P1-F checkpoint-10
inputs: fresh TSP heads (`sr`: the crop at w from frame t's periphery, lighting and predicted belief; `fwd`: from frame
t-1's periphery and lighting and frame t's predicted belief, i.e. through the action), P1-F's faithful loss,
identically seeded, the same frames, batches and budget; the arms differ only in the window trained per frame:
uniform; learnability (RHO-LOSS: current loss minus an irreducible-loss model's, the IL learner trained uniformly on
the held-out half); error (current loss); progress (gamma-progress, |EMA-copy loss - current loss|, Kim et al.'s
gamma 0.9995). Every non-uniform arm takes its arg-max with probability 0.8, a uniform window otherwise (MAGELLAN's
final epsilon 0.2). Labels select evaluation rows, label windows and describe the selections; they never score.
Usage (project venv, GPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.selection OUT_DIR [--smoke]
"""
import copy, importlib, json, math, os, sys, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

F = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.locus")
LS = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.landscape")
DEV, ROOT = F.DEV, TR.ROOT
CKPT = ROOT / "artifacts/experiments/20261030_p1_guided/run/ckpt_10.pt"
CEN = F.CENTRES.to(DEV)                                                  # [49, 2]
ARMS, MODES = ("uniform", "learnability", "error", "progress"), ("sr", "fwd")
NEED_ARMS = ("need", "need_learnability")                                # H2 (PLAN H2): need = 1 + contingency / its mean
STEPS, BATCH, WARMUP, SEED, EPS, GAMMA = 6000, 512, 250, 0, 0.2, 0.9995
EVALS = (0, 250, 500, 1000, 2000, 3000, 4500, 6000)


@torch.no_grad()
def need_of(model, cfg, enc, e, gen):
    """H2's need per window for frames 1..T-1 of episode e: 1 + F's action contingency of frame t (||D(a_{t-1}) -
    D(NOOP)||^2 over each window's cells and channels, D the clean estimate at production's first evaluation from pure
    noise, the 8 recorded context frames, the belief recomputed per action, one draw common to both actions) over its
    mean across the 49 windows; 1 where t < 8 (no context). Returns ([T - 1, 49], belief replay worst relative L2)."""
    T = len(enc[e]["p"])
    out, rows = torch.ones(T - 1, 49), [(e, t) for t in range(F.P, T)]
    if not rows:
        return out, 0.0
    bps, worst = F.beliefs(model, cfg, enc, rows)
    den, s, sd = model.denoiser, cfg["edm"]["sampler"], cfg["edm"]["sigma_data"]
    clamp, win = cfg["perception"]["periphery"]["tanh_clamp"], F.WIN.to(DEV)
    hat = s["sigma_max"] * (1 + min(s["s_churn"] / s["steps"], math.sqrt(2) - 1))
    shapes = [tuple(enc[e][k].shape[1:]) for k in ("p", "g", "v")]
    for b0 in range(0, len(rows), 32):
        br = rows[b0:b0 + 32]
        n = len(br)
        take = lambda k, dt: torch.stack([enc[ee][k][t + dt] for ee, t in br]).to(DEV)
        acts = torch.stack([take("prev_action", 0).long(), torch.zeros(n, dtype=torch.long, device=DEV)], 1).reshape(-1)
        bp = torch.stack([bps[r] for r in br]).to(DEV)[torch.arange(n, device=DEV).repeat_interleave(2), acts]
        with TR.autocast(DEV, cfg):
            cache = F.context_cache(model, cfg, take, 1, gen)
            cache = {"t": cache["t"], "kv": [(k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)) for k, v in cache["kv"]]}
            x = tuple(hat * torch.randn((n, 1, *shp), device=DEV, generator=gen).repeat_interleave(2, 0) for shp in shapes)
            Dd, _ = M.denoised(den, x, torch.full((2 * n, 1), hat, device=DEV), acts[:, None], model.belief.memory(bp)[:, None],
                               take("fix", 0).long().repeat_interleave(2, 0)[:, None], cache, clamp, sd)
        d = Dd[0][:, 0].float().view(n, 2, *shapes[0])
        c = ((d[:, 0] - d[:, 1]) ** 2).sum(-1)[:, win].sum(-1)            # [n, 49]
        out[[t - 1 for _, t in br]] = (1 + c / c.mean(1, keepdim=True).clamp(min=1e-12)).cpu()
    return out, worst


def collect(model, cfg, L, need_eps=()):
    """Per frame t >= 1 of every episode: frame t's periphery / lighting, frame t-1's, frame t's predicted belief, the
    frame's pixels, episode index, light, faced screen cell, whether a_{t-1} was DO; H2's need for episodes need_eps."""
    enc, eps = S.encode(model, L, cfg), S.episodes(L)
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    out = {k: [] for k in ("p", "pp", "v", "vp", "bp", "frame", "ep", "light", "faced", "do", "need")}
    gen, worst = torch.Generator(device=DEV).manual_seed(2033), 0.0
    for e, r in enumerate(enc):
        nd, wst = need_of(model, cfg, enc, e, gen) if e in need_eps else (torch.ones(len(r["p"]) - 1, 49), 0.0)
        out["need"].append(nd); worst = max(worst, wst)
        T = len(r["p"])
        g = s_off[e] + np.arange(1, T)
        d = F.FC.DIRS[L["s_dir"][g - 1]]
        for k, x in (("p", r["p"][1:]), ("pp", r["p"][:-1]), ("v", r["v"][1:]), ("vp", r["v"][:-1]), ("bp", r["bp"][1:]),
                     ("frame", D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"][1:T]),
                     ("ep", torch.full((T - 1,), e)), ("light", torch.as_tensor(L["s_light"][g])),
                     ("faced", torch.as_tensor((3 + d[:, 0]) * 9 + 4 + d[:, 1])),
                     ("do", torch.as_tensor(L["t_action"][a_off[e] + np.arange(T - 1)] == F.FC.DO))):
            out[k].append(torch.as_tensor(x))
    if worst > 0.02:
        raise RuntimeError(f"belief replay differs: {worst}")
    return {**{k: torch.cat(v) for k, v in out.items()}, "need_belief_replay_worst": worst}


def learner(cfg):
    torch.manual_seed(SEED)
    tsp, pixd = M.TSP(cfg).to(DEV), M.PixelDecoder(cfg).to(DEV)
    return tsp, pixd, [p for m in (tsp.tsp, tsp.belief, tsp.cell, pixd) for p in m.parameters()]


def outputs(tsp, pixd, X, idx, w, mode):
    """Decoded mean / log-scale and the target pixels for rows idx at windows w ([n] or [n, k] window indices)."""
    fix = CEN[w]
    cells = M.window(fix)
    bp, p, v = X["bp"][idx].to(DEV).float(), X["p" if mode == "sr" else "pp"][idx].to(DEV).float(), X["v" if mode == "sr" else "vp"][idx].to(DEV).float()
    frames = X["frame"][idx].to(DEV)
    if w.ndim == 2:
        k = w.shape[1]
        bp, p, v, frames = (x[:, None].expand(-1, k, *x.shape[1:]) for x in (bp, p, v, frames))
    mean, ls = pixd(tsp.predict(bp, p, v, cells), v, cells)
    return mean.float(), ls.float(), M.cell_pixels(M.crop(frames, fix))


def loss_all(tsp, pixd, X, idx, mode, cfg):
    """Pixel MSE (decoded mean clamped to [-1, 1]) at every window of rows idx: [n, 49]."""
    out = []
    with torch.no_grad(), TR.autocast(DEV, cfg):
        for b in torch.as_tensor(idx).split(64):
            mean, _, pix = outputs(tsp, pixd, X, b, torch.arange(49, device=DEV).expand(len(b), -1), mode)
            out.append((mean.clamp(-1, 1) - (pix.float() / 127.5 - 1)).square().flatten(2).mean(-1).cpu())
    return torch.cat(out)


def train(arm, mode, X, train_idx, cfg, il=None, evalfn=None):
    tsp, pixd, params = learner(cfg)
    ema = (copy.deepcopy(tsp).eval(), copy.deepcopy(pixd).eval()) if arm == "progress" else None
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    gen = torch.Generator().manual_seed(SEED)                              # the same frames and coin flips in every arm
    log, sel = [], {k: 0.0 for k in ("n", "faced", "hud", "night", "do_faced", "do")}
    hudw = torch.as_tensor((F.WIN.numpy() >= 63).any(-1), device=DEV)
    for step in range(STEPS + 1):
        if evalfn and step in EVALS:
            log.append({"step": step, **evalfn(tsp, pixd)})
        if step == STEPS:
            break
        for gp in opt.param_groups:
            gp["lr"] = o["lr"] * min(1.0, (step + 1) / WARMUP)
        idx = train_idx[torch.randint(0, len(train_idx), (BATCH,), generator=gen)]
        coin = torch.rand(BATCH, generator=gen).to(DEV) < EPS
        w = torch.randint(0, 49, (BATCH,), generator=gen).to(DEV)
        if arm == "need":
            w = torch.where(coin, w, X["need"][idx].to(DEV).argmax(1))
        elif arm != "uniform":
            score = loss_all(tsp, pixd, X, idx, mode, cfg).to(DEV)
            if arm == "learnability":
                score = score - loss_all(*il, X, idx, mode, cfg).to(DEV)
            elif arm == "progress":
                score = (loss_all(*ema, X, idx, mode, cfg).to(DEV) - score).abs()
            elif arm == "need_learnability":
                score = X["need"][idx].to(DEV) * (score - loss_all(*il, X, idx, mode, cfg).to(DEV))
            w = torch.where(coin, w, score.argmax(1))
        fw = CEN[w]
        cover = (M.window(fw) == X["faced"][idx].to(DEV)[:, None]).any(1)
        dn = X["do"][idx].to(DEV)
        for k, x in (("n", torch.ones_like(cover)), ("faced", cover), ("hud", hudw[w]), ("night", X["light"][idx].to(DEV) < 0.5),
                     ("do_faced", cover & dn), ("do", dn)):
            sel[k] += float(x.float().sum())
        with TR.autocast(DEV, cfg):
            mean, ls, pix = outputs(tsp, pixd, X, idx, w, mode)
        loss = (mean - (pix.float() / 127.5 - 1)).square().mean() + M.pixel_nll(mean.detach(), ls, pix).mean()   # faithful
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, o["grad_clip"])
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for m_old, m_new in zip(ema, (tsp, pixd)):
                    for a, b in zip(m_old.parameters(), m_new.parameters()):
                        a.mul_(GAMMA).add_(b, alpha=1 - GAMMA)
    stats = {"faced": sel["faced"] / sel["n"], "hud": sel["hud"] / sel["n"], "night_frames": sel["night"] / sel["n"],
             "faced_given_do": sel["do_faced"] / max(sel["do"], 1)}
    return (tsp.eval(), pixd.eval()), log, stats


def summary(out_dir):
    """PLAN H's readings from the saved arrays (CPU): per learner, arm and evaluation step, held-out pixel MSE on (1)
    all windows of the random rows, (2) the DO classes' consequence windows, (3) random rows by light; paired
    differences against uniform (episode-bootstrap SEs); steps to reach uniform's final value (linear in log loss)."""
    lab = np.load(out_dir / "labels.npz")
    cls, ep, light, cons = lab["cls"], lab["ep"], lab["light"], lab["consequence"]
    rnd, do = cls == "random", np.isin(cls, list(F.ATTEMPTS) + ["stone_no_tool"])
    pops = {"random_all_windows": (rnd, None), "random_day": (rnd & (light >= 0.7), None), "random_night": (rnd & (light <= 0.3), None),
            "do_consequence_windows": (do, cons), "do_other_windows": (do, ~cons)}
    fz = np.load(Path(__file__).resolve().parent / "locus/p1f_10.npz")
    if len(fz["ep"]) == len(ep) and (fz["ep"] == ep).all() and (fz["t"] == lab["t"]).all():   # F's rows are G's rows
        c = fz["contingency"]
        nw = 1 + c / np.maximum(c.mean(1, keepdims=True), 1e-12)          # H2's need on the evaluation rows
        pops.update({"need_weighted_all_rows": (np.ones(len(ep), bool), nw), "need_weighted_do_rows": (do, nw)})
    def row_value(loss, m, w):
        return loss[m].mean(1) if w is None else (loss[m] * w[m]).sum(1) / np.maximum(w[m].sum(1), 1)
    res = {}
    ARMS = [a for a in ("uniform", "learnability", "error", "progress", *NEED_ARMS) if (out_dir / f"sr_{a}.npz").exists()]
    for mode in MODES:
        z = {arm: np.load(out_dir / f"{mode}_{arm}.npz") for arm in ARMS}
        steps = z["uniform"]["steps"]
        res[mode] = {}
        for pname, (m, w) in pops.items():
            r = res[mode][pname] = {"steps": steps.tolist()}
            for arm in ARMS:
                vals = [row_value(z[arm]["loss"][i], m, w) for i in range(len(steps))]
                base = [row_value(z["uniform"]["loss"][i], m, w) for i in range(len(steps))]
                r[arm] = {"mean": [float(v.mean()) for v in vals],
                          "minus_uniform": [LS.G.estimate(v - b, np.ones(len(v), bool), ep[m]) for v, b in zip(vals, base)]}
            final = r["uniform"]["mean"][-1]
            for arm in ARMS:
                y = np.log(np.array(r[arm]["mean"]))
                hit = np.nonzero(y <= np.log(final))[0]
                if len(hit) and hit[0] > 0:
                    i = hit[0]; f = (np.log(final) - y[i - 1]) / (y[i] - y[i - 1])
                    r[arm]["steps_to_uniform_final"] = float(steps[i - 1] + f * (steps[i] - steps[i - 1]))
                else:
                    r[arm]["steps_to_uniform_final"] = None
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1, default=float))
    for mode, rr in res.items():
        for pname, r in rr.items():
            print(f"== {mode} {pname}")
            for arm in ARMS:
                d = r[arm]["minus_uniform"]
                print(f"   {arm:12s} final {r[arm]['mean'][-1]:.5f} to-uniform-final {r[arm]['steps_to_uniform_final']}",
                      " ".join(f"{s}:{x['est']:+.5f}({x['se']:.5f})" for s, x in zip(r["steps"], d) if s in (1000, 2000, 3000, 6000)))
    return res


def main():
    global STEPS, EVALS
    if sys.argv[1] == "--summary":
        return summary(Path(sys.argv[2]))
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        STEPS, EVALS = 40, (0, 20, 40)
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    Ltr = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz")), 20 if smoke else 400)
    Ldv = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12 if smoke else F.N_EP)
    arms = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--arms=")), ARMS)
    X = collect(model, cfg, Ltr, need_eps=set(range(0, len(Ltr["ep_T"]), 2)) if "--need" in sys.argv else ())
    ctx, face, d = F.contexts(Ldv, np.random.default_rng(2031))            # G's (F's) evaluation rows
    if smoke:
        ctx = {c: r[:6] for c, r in ctx.items() if r}
    rows = [r for c in ctx for r in ctx[c]]
    E_all = collect(model, cfg, Ldv)
    s_off = np.r_[0, np.cumsum(Ldv["ep_T"])]                               # collect drops frame 0 of each episode
    eval_idx = torch.as_tensor([s_off[e] + t - 1 for e, t in rows])
    lab = LS.window_labels(Ldv, rows, ctx, d)
    np.savez_compressed(out_dir / "labels.npz", **lab)
    del model
    torch.cuda.empty_cache()
    held = (X["ep"] % 2 == 1).nonzero()[:, 0]                              # the IL learner's half (RHO's holdout split)
    fit = (X["ep"] % 2 == 0).nonzero()[:, 0]
    res = {"checkpoint": str(CKPT), "frames": {"fit": len(fit), "il": len(held)}, "eval_rows": len(rows), "arms": list(arms),
           "need_belief_replay_worst": X["need_belief_replay_worst"], "runs": {}}
    for mode in MODES:
        evalfn = lambda tsp, pixd: {"loss": loss_all(tsp, pixd, E_all, eval_idx, mode, cfg).numpy()}
        t0 = time.time()
        il, _, _ = train("uniform", mode, X, held, cfg)
        res["runs"][f"{mode}_il"] = {"seconds": time.time() - t0}
        il_eval = loss_all(*il, E_all, eval_idx, mode, cfg).numpy()
        np.save(out_dir / f"{mode}_il_eval.npy", il_eval)
        for arm in arms:
            t0 = time.time()
            _, log, stats = train(arm, mode, X, fit, cfg, il=il, evalfn=evalfn)
            np.savez_compressed(out_dir / f"{mode}_{arm}.npz", steps=np.array([x["step"] for x in log]),
                                loss=np.stack([x["loss"] for x in log]))
            res["runs"][f"{mode}_{arm}"] = {"seconds": time.time() - t0, "selection": stats,
                                            "eval_mean": [float(x["loss"].mean()) for x in log]}
            print(mode, arm, round(time.time() - t0), "s", json.dumps(res["runs"][f"{mode}_{arm}"]), flush=True)
            (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
