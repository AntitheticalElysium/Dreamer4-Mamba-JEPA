"""Check 4 and two questions (user, 10-30), P1 world 1 checkpoint 10, real tokens, random-prior fixations (S1-seeded,
P1's training distribution), dev (first 300 episodes) for readings:
A. IS TSP AFFECTED? TSP and the reconstruction share PixelDecoder: the mean head reads each path's own feature, the
   log-scale reads only (lighting, cell), so the two paths have the SAME scale by construction. Measured per path:
   does the digit reach the path's feature (MLP probe with slot one-hot, episode-parity split), does the decoded mean
   draw the right value (nearest own-cell template) and the right icon (nearest over all counters' templates, frames
   showing the icon only), and
   the per-pixel NLL gradient on digit pixels vs constant HUD pixels vs map pixels.
B. BEYOND INCREMENTS: per content (each HUD counter; each block type and each mob kind on map cells, day frames;
   all map cells day / night), the decoded mean's absolute error (0-255) and NLL per pixel, both paths.
C. MATCHED REFIT (Codex's control): a fresh reconstruction adapter (TSP.rec + its cell embedding) and PixelDecoder,
   identically seeded, trained on frozen glimpse tokens of train-400 with the same data, batches, budget and
   optimizer under four losses: nll (the current discretised logistic NLL), beta_nll (Seitzer et al.: NLL x
   stop-gradient(scale), beta 0.5), faithful (Stirn et al. in spirit: the mean by squared error, the scale by NLL on
   the detached mean), mse (the mean only). Read with A / B's reconstruction metrics. Claim: the loss's learned
   per-position scale causes the digit-blindness; competing: the decoder cannot render digits from these tokens at
   this budget whatever the loss (then every arm fails), or the trap needs the joint training's early dynamics
   (then nll renders digits here too).
D. (--tsp, user 10-31) TSP's own path: a fresh TSP adapter (tsp, belief projection, cell embedding) + PixelDecoder
   trained through TSP only under nll / beta_nll / faithful on frozen checkpoint-10 inputs (predicted belief, periphery,
   lighting), read on TSP's path: does TSP prefer an uncertainty-weighted loss (its target is genuinely uncertain
   before the glimpse)?
Usage (project venv, GPU): python -m artifacts.experiments.20261030_decoder_diagnosis.refit OUT_JSON [--smoke] [--tsp]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch
import craftax.craftax_classic.constants as C

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

CH = importlib.import_module("artifacts.experiments.20261030_decoder_diagnosis.checks")
IN, DEV, ROOT = CH.IN, CH.DEV, CH.ROOT
CKPT = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
BLOCKS = {b.value: b.name.lower() for b in C.BlockType}
MOBS = {k: n for k, n in enumerate(G.KINDS, 1)}
N_EVAL_EP, STEPS, BATCH, WARMUP, SEED = 300, 6000, 512, 250, 0
ARMS = ("nll", "beta_nll", "faithful", "mse")


def collect(model, cfg, L, n_ep, tsp_inputs):
    """Per frame: window glimpse tokens [n, 9, 64], lighting, window cells, crop pixels [n, 9, 147], and per window
    cell its content: block id (map), mob kind (map, 0 none), counter value (HUD counters, else -1); night flag;
    episode; with tsp_inputs also the periphery tokens and the predicted belief."""
    L = S.first_episodes(dict(L), n_ep) if n_ep else L
    enc = S.encode(model, L, cfg)
    grids, eps = G.episode_grids(L), S.episodes(L)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = {k: [] for k in ("g", "v", "cells", "pix", "block", "mob", "val", "night", "ep") + (("p", "bp") if tsp_inputs else ())}
    for e, r in enumerate(enc):
        fix = r["fix"]
        cells = M.window(fix).numpy()
        ob = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"]
        pix = M.cell_pixels(M.crop(ob, fix))
        st = np.arange(s_off[e], s_off[e + 1])
        rr, cc = np.minimum(cells, 62) // 9, np.minimum(cells, 62) % 9
        tt = np.arange(len(st))[:, None]
        block = np.where(cells < 63, L["s_view"][st[:, None], rr, cc], -1)
        mob = np.where(cells < 63, grids[e]["grid"][tt, rr, cc], -1)
        val = np.full(cells.shape, -1)
        for c, (arr, col) in CH.COUNTERS.items():
            m = cells == c
            val[m] = np.clip(L[arr][st, col], 0, None)[np.nonzero(m)[0]]
        for k, x in (("g", r["g"]), ("v", r["v"]), ("cells", torch.as_tensor(cells)), ("pix", pix),
                     ("block", torch.as_tensor(block)), ("mob", torch.as_tensor(mob)), ("val", torch.as_tensor(val)),
                     ("night", torch.as_tensor(L["s_light"][st] < 0.5)), ("ep", torch.full((len(st),), e))):
            out[k].append(x)
        if tsp_inputs:
            out["p"].append(r["p"]); out["bp"].append(r["bp"])
    return {k: torch.cat(v) for k, v in out.items()}


def decode(tsp, pixd, E, idx, path):
    """(mean in [-1, 1] [b, 9, 147], log-scale [b, 9, 147], path feature [b, 9, hidden]) for rows idx."""
    g, v, cells = E["g"][idx].float().to(DEV), E["v"][idx].float().to(DEV), E["cells"][idx].to(DEV)
    f = tsp.reconstruct(g, v, cells) if path == "rec" else tsp.predict(E["bp"][idx].float().to(DEV), E["p"][idx].float().to(DEV), v, cells)
    mean, ls = pixd(f, v, cells)
    return mean.float(), ls.float(), f


def evaluate(tsp, pixd, E, path, tpl, cfg, grads=True):
    """Section A / B metrics of one decoding path on the eval set."""
    allT = [(c, int(a), t) for c, (vals, T) in tpl.items() for a, t in zip(vals, T)]
    Tall = torch.stack([t for _, _, t in allT]).to(DEV)
    owner = np.array([c for c, _, _ in allT])
    digit = {c: (T[torch.as_tensor(vals >= 1)].std(0) > 0).to(DEV) for c, (vals, T) in tpl.items()}
    acc = {}

    def add(key, field, x, n=1):
        d = acc.setdefault(key, {})
        d[field] = d.get(field, 0.0) + float(x)
        d[field + "_n"] = d.get(field + "_n", 0) + n
    for b in torch.arange(len(E["g"])).split(4096):
        with torch.no_grad(), TR.autocast(DEV, cfg):
            mean, ls, _ = decode(tsp, pixd, E, b, path)
        pix = E["pix"][b].to(DEV)
        y = pix.float() / 127.5 - 1
        mu = mean.detach().clamp(-1, 1).requires_grad_(grads)
        nll = M.pixel_nll(mu, ls, pix)                                          # [b, 9, 147]
        if grads:
            nll.sum().backward()
            gr = mu.grad.abs()
        mae = (y - mu.detach()).abs() * 127.5
        cells, block, mob, val = (E[k][b].to(DEV) for k in ("cells", "block", "mob", "val"))
        night = E["night"][b].to(DEV)[:, None].expand_as(cells)
        dec = ((mu.detach() + 1) * 127.5)
        for c, (vals, T) in tpl.items():
            m = cells == c
            if not m.any():
                continue
            name = CH.NAMES[c]
            d = dec[m]
            own = torch.as_tensor(vals, device=DEV)[torch.cdist(d, T.to(DEV)).argmin(1)]
            add(f"hud_{name}", "digit_correct", (own == val[m]).sum(), int(m.sum()))
            on = val[m] >= 1                                                    # value 0 is the same empty cell for every counter
            if on.any():
                add(f"hud_{name}", "icon_correct", (torch.as_tensor(owner, device=DEV)[torch.cdist(d[on], Tall).argmin(1)] == c).sum(), int(on.sum()))
            add(f"hud_{name}", "mae", mae[m].mean(1).sum(), int(m.sum())); add(f"hud_{name}", "nll", nll[m].detach().mean(1).sum(), int(m.sum()))
            if grads:
                add("grad_hud_digit_pixels", "grad", gr[m][:, digit[c]].sum(), int(m.sum()) * int(digit[c].sum()))
                add("grad_hud_constant_pixels", "grad", gr[m][:, ~digit[c]].sum(), int(m.sum()) * int((~digit[c]).sum()))
        mapc = cells < 63
        if grads:
            add("grad_map_pixels", "grad", gr[mapc].sum(), int(mapc.sum()) * 147)
        for nm, m in (("map_day", mapc & ~night), ("map_night", mapc & night)):
            add(nm, "mae", mae[m].mean(1).sum(), int(m.sum())); add(nm, "nll", nll[m].detach().mean(1).sum(), int(m.sum()))
        for bid in torch.unique(block[mapc & ~night & (mob == 0)]).tolist():
            m = mapc & ~night & (mob == 0) & (block == bid)
            add(f"block_{BLOCKS.get(bid, bid)}", "mae", mae[m].mean(1).sum(), int(m.sum()))
            add(f"block_{BLOCKS.get(bid, bid)}", "nll", nll[m].detach().mean(1).sum(), int(m.sum()))
        for k, nm in MOBS.items():
            m = mapc & ~night & (mob == k)
            if m.any():
                add(f"mob_{nm}", "mae", mae[m].mean(1).sum(), int(m.sum())); add(f"mob_{nm}", "nll", nll[m].detach().mean(1).sum(), int(m.sum()))
    res = {}
    for key, d in acc.items():
        res[key] = {f: d[f] / max(d[f + "_n"], 1) for f in d if not f.endswith("_n")} | {"n": max(v for f, v in d.items() if f.endswith("_n"))}
    with torch.no_grad():
        vv = E["v"][:4096].float().to(DEV)
        cl = torch.arange(81, device=DEV)
        lsc = pixd.scale(torch.cat([vv[:, None].expand(-1, 81, -1), pixd.cell(cl)[None].expand(len(vv), -1, -1)], -1)).float().mean((0, 2))
    res["log_scale"] = {"hud_body": float(lsc[63:67].mean()), "hud_inventory": float(lsc[67:70].mean()), "map": float(lsc[:63].mean())}
    return res


def probe_digit(tsp, pixd, E, path, cfg):
    """Digit of drink / wood / health from the path's feature at the counter's slot (+ slot one-hot): MLP probe,
    even episodes -> odd episodes."""
    out = {}
    for c in (65, 68, 63):
        rows, slots = (E["cells"] == c).nonzero(as_tuple=True)
        if len(rows) < 500:
            continue
        feats = []
        for b in torch.arange(len(rows)).split(4096):
            with torch.no_grad(), TR.autocast(DEV, cfg):
                _, _, f = decode(tsp, pixd, E, rows[b], path)
            feats.append(f[torch.arange(len(b)), slots[b].to(DEV)].float().cpu())
        X = torch.cat([torch.cat(feats), torch.nn.functional.one_hot(slots, 9).float()], 1)
        y = E["val"][rows, slots].numpy()
        tr = (E["ep"][rows] % 2 == 0).numpy()
        r = IN.fit_mlp(X[torch.as_tensor(tr)], y[tr])
        ok = ~tr & np.isin(y, r["classes"])
        out[CH.NAMES[c]] = {"mlp_accuracy": float((IN.predict(r, X[torch.as_tensor(ok)]) == y[ok]).mean()), "rows": int(len(rows))}
    return out


def refit(arm, F, cfg, path="rec"):
    """A fresh adapter (reconstruction, or TSP's own) + PixelDecoder, seeded identically for every arm, on frozen
    inputs; trained through one path only."""
    torch.manual_seed(SEED)
    tsp, pixd = M.TSP(cfg).to(DEV), M.PixelDecoder(cfg).to(DEV)
    adapter = [tsp.rec] if path == "rec" else [tsp.tsp, tsp.belief]
    params = [p for m in adapter + [tsp.cell, pixd] for p in m.parameters()]
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(params, lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    log = []
    for step in range(STEPS):
        for gp in opt.param_groups:
            gp["lr"] = o["lr"] * min(1.0, (step + 1) / WARMUP)
        idx = torch.randint(0, len(F["g"]), (BATCH,), generator=gen)
        with TR.autocast(DEV, cfg):
            mean, ls, _ = decode(tsp, pixd, F, idx, path)
        pix = F["pix"][idx].to(DEV)
        y = pix.float() / 127.5 - 1
        if arm == "nll":
            loss = M.pixel_nll(mean, ls, pix).mean()
        elif arm == "beta_nll":
            loss = (M.pixel_nll(mean, ls, pix) * ls.clamp(min=-7).exp().detach()).mean()
        elif arm == "faithful":
            loss = (mean - y).square().mean() + M.pixel_nll(mean.detach(), ls, pix).mean()
        else:
            loss = (mean - y).square().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, o["grad_clip"])
        opt.step()
        if step % 1000 == 0 or step == STEPS - 1:
            log.append({"step": step, "loss": float(loss.detach())})
    tsp.eval(); pixd.eval()
    return tsp, pixd, log


def main():
    global N_EVAL_EP, STEPS
    out_path = Path(sys.argv[1])
    smoke = "--smoke" in sys.argv
    if smoke:
        N_EVAL_EP, STEPS = 8, 50
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    Ltr = dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz"))
    tpl = CH.templates(L)
    E = collect(model, cfg, L, N_EVAL_EP, True)
    print("eval frames", len(E["g"]), flush=True)
    if "--tsp" in sys.argv:                                              # TSP's own path (user, 10-31), its own output file
        F = collect(model, cfg, Ltr, 20 if smoke else 0, True)
        F = {k: (v.to(DEV) if k in ("g", "v", "cells", "pix", "p", "bp") else v) for k, v in F.items()}
        del model
        torch.cuda.empty_cache()
        res = {"checkpoint": str(CKPT), "eval_frames": int(len(E["g"])), "fit_frames": int(len(F["g"])), "refit_tsp": {}}
        for arm in ("nll", "beta_nll", "faithful"):
            tsp, pixd, log = refit(arm, F, cfg, "tsp")
            res["refit_tsp"][arm] = {"log": log, **evaluate(tsp, pixd, E, "tsp", tpl, cfg)}
            print("refit_tsp", arm, json.dumps({k: v for k, v in res["refit_tsp"][arm].items() if k.startswith(("hud", "log_scale", "map", "mob"))}), flush=True)
            out_path.write_text(json.dumps(res, indent=1, default=str))
        return
    res = {"checkpoint": str(CKPT), "eval_frames": int(len(E["g"])), "trained": {}, "refit": {}}
    for path in ("rec", "tsp"):
        res["trained"][path] = evaluate(model.tsp, model.pixels, E, path, tpl, cfg)
        res["trained"][path]["probe"] = probe_digit(model.tsp, model.pixels, E, path, cfg)
        print("trained", path, json.dumps({k: v for k, v in res["trained"][path].items() if k.startswith(("hud", "grad", "probe", "log"))}), flush=True)
    out_path.write_text(json.dumps(res, indent=1, default=str))
    F = collect(model, cfg, Ltr, 20 if smoke else 0, False)
    F = {k: (v.to(DEV) if k in ("g", "v", "cells", "pix") else v) for k, v in F.items()}
    del model
    torch.cuda.empty_cache()
    print("fit frames", len(F["g"]), flush=True)
    for arm in ARMS:
        tsp, pixd, log = refit(arm, F, cfg)
        res["refit"][arm] = {"log": log, **evaluate(tsp, pixd, E, "rec", tpl, cfg)}
        print("refit", arm, json.dumps({k: v for k, v in res["refit"][arm].items() if k.startswith(("hud", "log_scale", "map", "block_table", "mob"))}), flush=True)
        out_path.write_text(json.dumps(res, indent=1, default=str))
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
