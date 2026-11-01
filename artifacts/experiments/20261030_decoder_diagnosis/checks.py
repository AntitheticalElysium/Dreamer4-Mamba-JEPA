"""Why the shared pixel decoder is digit-blind (user, 10-30; DISSECTION of M0.5). Three cheap checks, GPU, dev and
train-400, real tokens only (no generation):
1. INFORMATION FLOW. At the rule's dominant windows (drink 65 at centre (7, 2), wood 68 at (7, 5)), the counter's value
   probed from (a) the cell's glimpse token, (b) the reconstruction adapter's output (TSP.reconstruct, the feature the
   decoder's mean reads), (c) the decoder's mean pixels; MLP (instruments.fit_mlp) and linear probes fitted on
   train-400, read on dev; (c) also by the nearest exact HUD template. Claim: the digit reaches the adapter's output
   and dies in the mean head (a training-pressure failure); competing: the adapter already drops it.
2. GRADIENT. The reconstruction NLL's gradient with respect to the decoder mean, per pixel, on real crops: HUD digit
   pixels (pixels whose value differs across the cell's value templates), HUD constant pixels (icon, background), map
   pixels; beside the scale 1/s and the error |y - mu|. Claim (derivation): a badly wrong pixel's gradient saturates at
   1/s, so the wide HUD scale weakens the pull on digit pixels.
3. TRAJECTORY. P1 world 1 checkpoints 1-10: the decoder's log-scale on HUD body stats (63-66), inventory (67-69) and map
   cells, at each checkpoint's own lighting latents, and the decoded drink / wood cell's nearest real template.
   Claim: the HUD scale widened while the mean failed early and stayed (a stable trap); competing: wide from the start.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_decoder_diagnosis.checks OUT_JSON
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
IN = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.instruments")
PM = EV.PM
DEV = EV.DEV
ROOT = EV.ROOT
P1 = ROOT / "artifacts/saccade_m0/p1_world1"
TARGETS = {"drink": (65, (7, 2)), "wood": (68, (7, 5))}
NAMES = {63: "health", 64: "food", 65: "drink", 66: "energy", 67: "sapling", 68: "wood", 69: "stone"}
COUNTERS = {63: ("s_body", 0), 64: ("s_body", 1), 65: ("s_body", 2), 66: ("s_body", 3), 67: ("s_inv", 5),
            68: ("s_inv", 0), 69: ("s_inv", 1)}                            # pixel-checked 10-30 (exact per value)
BODY, INV = [63, 64, 65, 66], [67, 68, 69]


def templates(L, n_ep=300):
    """Exact pixel template per HUD counter cell and value (the cells render exactly per value): {cell: (values, [k, 147])}."""
    eps, s_off = S.episodes(L), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    acc = {c: {} for c in COUNTERS}
    for e in range(min(n_ep, len(eps))):
        ob = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"]
        for c, (arr, col) in COUNTERS.items():
            r, k = divmod(c, 9)
            val = np.clip(L[arr][s_off[e]:s_off[e + 1], col], 0, None)
            for v in np.unique(val):
                if int(v) not in acc[c]:
                    acc[c][int(v)] = ob[int(np.nonzero(val == v)[0][0]), 7 * r:7 * r + 7, 7 * k:7 * k + 7].reshape(-1).float()
    return {c: (np.array(sorted(a)), torch.stack([a[v] for v in sorted(a)])) for c, a in acc.items()}


def load(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    model = M.M0(ck["cfg"]).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    return model, ck["cfg"]


def rows(enc, L, cell, centre):
    """Frames fixated at `centre`: window tokens [n, 9, 64], lighting [n, dv], state index [n], the cell's slot."""
    s = IN.states(enc, L)
    fix = torch.cat([r["fix"] for r in enc])
    m = ((fix[:, 0] == centre[0]) & (fix[:, 1] == centre[1])).numpy()
    win = M.window(torch.as_tensor(centre))
    slot = int((win == cell).nonzero()[0, 0])
    g, v = torch.cat([r["g"] for r in enc])[torch.as_tensor(m)], torch.cat([r["v"] for r in enc])[torch.as_tensor(m)]
    return g, v, s[m], slot, win


@torch.no_grad()
def decoder_parts(model, cfg, g, v, win, slot):
    """(adapter feature of the slot [n, hidden], decoder mean [n, 147] uint8 scale, log-scale [n, 147])."""
    feats, means, scales = [], [], []
    for a in range(0, len(g), 2048):
        gg, vv = g[a:a + 2048].float().to(DEV), v[a:a + 2048].float().to(DEV)
        cells = win.to(DEV).expand(len(gg), 9)
        with TR.autocast(DEV, cfg):
            f = model.tsp.reconstruct(gg, vv, cells)
            mean, ls = model.pixels(f, vv, cells)
        feats.append(f[:, slot].float().cpu()); means.append(((mean[:, slot].float().clamp(-1, 1) + 1) * 127.5).cpu())
        scales.append(ls[:, slot].float().cpu())
    return torch.cat(feats), torch.cat(means), torch.cat(scales)


def linear_probe(Xtr, ytr, Xdv, ydv, steps=2000):
    """Multinomial logistic regression (AdamW, standardised inputs): dev accuracy."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    classes = np.unique(ytr)
    W = torch.zeros(Xtr.shape[1], len(classes), requires_grad=True, device=DEV)
    b = torch.zeros(len(classes), requires_grad=True, device=DEV)
    opt = torch.optim.AdamW([W, b], lr=1e-2, weight_decay=1e-4)
    Xt, yt = ((Xtr - mu) / sd).to(DEV), torch.as_tensor(np.searchsorted(classes, ytr), device=DEV)
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad(); torch.nn.functional.cross_entropy(Xt @ W + b, yt).backward(); opt.step()
    ok = np.isin(ydv, classes)
    pred = classes[(((Xdv - mu) / sd).to(DEV) @ W + b).argmax(1).cpu().numpy()]
    return float((pred[ok] == ydv[ok]).mean())


def check_information(model, cfg, L, Ltr, enc, enc_tr, tpl):
    out = {}
    for name, (cell, centre) in TARGETS.items():
        gt, vt, st, slot, win = rows(enc_tr, Ltr, cell, centre)
        gd, vd, sd_, _, _ = rows(enc, L, cell, centre)
        yt, yd = EV.hud_value(Ltr, name, st), EV.hud_value(L, name, sd_)
        ft, mt, _ = decoder_parts(model, cfg, gt, vt, win, slot)
        fd, md, _ = decoder_parts(model, cfg, gd, vd, win, slot)
        reps = {"glimpse_token": (gt[:, slot].float(), gd[:, slot].float()), "adapter_output": (ft, fd),
                "decoder_mean_pixels": (mt / 255, md / 255)}
        res = {"train_rows": int(len(yt)), "dev_rows": int(len(yd))}
        for k, (Xt, Xd) in reps.items():
            r = IN.fit_mlp(Xt, yt)
            ok = np.isin(yd, r["classes"])
            res[k] = {"mlp_dev_accuracy": float((IN.predict(r, Xd[torch.as_tensor(ok)]) == yd[ok]).mean()),
                      "linear_dev_accuracy": linear_probe(Xt, yt, Xd, yd)}
        vals, T = tpl[cell]
        pv = vals[torch.cdist(md.float(), T).argmin(1).numpy()]
        res["decoder_mean_nearest_template_accuracy"] = float((pv == yd).mean())
        res["decoder_mean_nearest_template_values"] = {int(a): int(b) for a, b in zip(*np.unique(pv, return_counts=True))}
        out[name] = res
        print("information", name, json.dumps(res), flush=True)
    return out


def check_gradient(model, cfg, L, enc, tpl):
    """Real crops of the drink / wood windows: per-pixel d NLL / d mean, 1 / s, |y - mu| by pixel group."""
    eps = S.episodes(L)
    out = {}
    for name, (cell, centre) in TARGETS.items():
        g, v, st, slot, win = rows(enc, L, cell, centre)
        n = min(3000, len(st))
        pick = np.random.default_rng(0).choice(len(st), n, replace=False)
        s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
        ep = np.searchsorted(s_off, st[pick], side="right") - 1
        obs = {int(e): D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"] for e in np.unique(ep)}
        frames = torch.stack([obs[int(e)][int(s - s_off[e])] for e, s in zip(ep, st[pick])])
        fx = torch.as_tensor(centre).expand(n, 2)
        pix = M.cell_pixels(M.crop(frames, fx)).to(DEV)                           # [n, 9, 147]
        cells = win.to(DEV).expand(n, 9)
        gg, vv = g[pick].float().to(DEV), v[pick].float().to(DEV)
        with TR.autocast(DEV, cfg), torch.no_grad():
            mean, ls = model.pixels(model.tsp.reconstruct(gg, vv, cells), vv, cells)
        mean = mean.float().detach().requires_grad_(True)
        M.pixel_nll(mean, ls.float(), pix).sum().backward()
        grad, s_inv = mean.grad.abs().cpu(), torch.exp(-ls.float().clamp(min=-7)).cpu()
        err = ((pix.float() / 127.5 - 1) - mean.detach()).abs().cpu()
        vals, T = tpl[cell]
        digit = T[vals >= 1].std(0) > 0                                           # pixels that change with the value
        groups = {"hud_digit_pixels": (slot, digit), "hud_constant_pixels": (slot, ~digit)}
        res = {}
        for gname, (sl, mask) in groups.items():
            res[gname] = {"grad": float(grad[:, sl, mask].mean()), "inv_scale": float(s_inv[:, sl, mask].mean()),
                          "abs_error": float(err[:, sl, mask].mean()), "pixels": int(mask.sum())}
        map_slots = [j for j, c in enumerate(win.tolist()) if c < 63]
        res["map_pixels_same_windows"] = {"grad": float(grad[:, map_slots].mean()), "inv_scale": float(s_inv[:, map_slots].mean()),
                                          "abs_error": float(err[:, map_slots].mean())}
        out[name] = res
        print("gradient", name, json.dumps(res), flush=True)
    return out


def check_trajectory(L, rule, tpl):
    out = {}
    L30 = S.first_episodes(dict(L), 60)
    W = PM.window_of_state(L30, rule)
    for k in range(1, 11):
        model, cfg = load(P1 / f"ckpt_{k:02d}.pt")
        enc = EV.encode_with(model, L30, cfg, W)
        v = torch.cat([r["v"] for r in enc]).float()
        vv = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(0))[:2048]].to(DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            cells = torch.arange(81, device=DEV)
            ls = model.pixels.scale(torch.cat([vv[:, None].expand(-1, 81, -1), model.pixels.cell(cells)[None].expand(len(vv), -1, -1)], -1))
        ls = ls.float().mean((0, 2)).cpu()
        row = {"log_scale": {"body": float(ls[BODY].mean()), "inventory": float(ls[INV].mean()), "map": float(ls[:63].mean()),
                             **{NAMES[c]: float(ls[c]) for c in BODY + INV}}}
        for name, (cell, centre) in TARGETS.items():
            g, vg, st, slot, win = rows(enc, L30, cell, centre)
            if len(st) == 0:
                continue
            _, md, _ = decoder_parts(model, cfg, g, vg, win, slot)
            vals_all = []
            for c in BODY + INV:
                vals_all += [(c, int(a), t) for a, t in zip(*tpl[c])]
            Tall = torch.stack([t for _, _, t in vals_all])
            near = torch.cdist(md.float(), Tall).argmin(1).numpy()
            lab = [f"{NAMES[vals_all[j][0]]}={vals_all[j][1]}" for j in near]
            u, c = np.unique(lab, return_counts=True)
            truth = EV.hud_value(L30, name, st)
            row[f"decoded_{name}"] = {"top": {str(a): int(b) for a, b in sorted(zip(u, c), key=lambda x: -x[1])[:3]},
                                      "own_cell_correct_value": float(np.mean([l == f"{name}={t}" for l, t in zip(lab, truth)]))}
        out[k] = row
        print("trajectory", k, json.dumps(row), flush=True)
        del model, enc
        torch.cuda.empty_cache()
    return out


def main():
    out_path = Path(sys.argv[1])
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    Ltr = dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz"))
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    tpl = templates(L)
    res = {"checkpoint": str(P1 / "ckpt_10.pt")}
    model, cfg = load(P1 / "ckpt_10.pt")
    enc = EV.encode_with(model, L, cfg, PM.window_of_state(L, rule))
    enc_tr = EV.encode_with(model, Ltr, cfg, PM.window_of_state(Ltr, rule))
    res["information"] = check_information(model, cfg, L, Ltr, enc, enc_tr, tpl)
    res["gradient"] = check_gradient(model, cfg, L, enc, tpl)
    del model, enc, enc_tr
    torch.cuda.empty_cache()
    res["trajectory"] = check_trajectory(L, rule, tpl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=1))
    print("saved", out_path, flush=True)


if __name__ == "__main__":
    main()
