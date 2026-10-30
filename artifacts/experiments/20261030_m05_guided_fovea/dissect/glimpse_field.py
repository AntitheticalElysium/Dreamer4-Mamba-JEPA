"""Dissection of M0.5 arm G, GPU part (user, 10-30: what exactly fails, beyond the class rates). Guided route, dev,
evaluate.py's rows, seeds and saved readers (`eval/LABEL_readers.pt`), per checkpoint:
1. INSTRUMENT on generated tokens (the pixel route turned out digit-blind: the fovea reconstruction's decoder mean
   renders a HUD cell as its icon with a "1" whatever the value, real tokens included, so it reads only icon
   presence; reported as such). The production samples are regenerated with the evaluation's own calls and seeds
   (catalogue.run / field_calibration.run, keep=True) and checked against the saved readings. Each readable sample's
   distance to its read value's real mean, over that value's support radius, beside the same ratio for the rows' real
   tokens, and whether the token also lies inside the end / start value's support (ambiguity); and the pixel route: the glimpse tokens decoded by the fovea reconstruction (decoder mean)
   and assigned to the nearest exact HUD template (the HUD cells render exactly per value; dev frames), qualified on
   the rows' real tokens.
2. PROGRESS below the sampling threshold. The denoiser's clean estimate of the glimpse at the consequence cell from
   the real target frame noised at sigma (catalogue.run's field, glimpse instead of periphery), projected on the
   reader's real means: f_up = <D - mu_start, mu_end - mu_start> / |mu_end - mu_start|^2 (1 = the event, 0 = copy),
   f_down the same toward start - 1. For a deterministic event a learned field has f_up -> 1 at every sigma.
3. ACTION and the change's direction. evaluate.action_routing's draws (same seeds) with the generated values kept:
   up / copy / down / far under the recomputed belief and under the NOOP belief.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.dissect.glimpse_field [LABEL ...]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

E = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.evaluate")
EV, IN, IP, CAT, FC, PM = E.EV, E.IN, E.IP, E.CAT, E.FC, E.PM
DEV, P = E.DEV, E.P
HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
FT = "artifacts/experiments/20261030_m05_guided_fovea/finetune/guided"
CKPTS = {"base": "artifacts/saccade_m0/p1_world1/ckpt_10.pt", "guided_0300": f"{FT}/ft_0300.pt",
         "guided_0750": f"{FT}/ft_0750.pt", "guided_1500": f"{FT}/ft_1500.pt"}
KINDS = ("drink", "wood", "drink_fail", "risky")
FIELD_KINDS = ("drink", "wood")
FIELD_SIGMAS = (0.377, 1.2, 3.5, 6.67, 20.0)
HUD_CELL = {"drink": 65, "wood": 68, "health": 63}
IMG_PER_CAT = 6


def readers(label):
    """evaluate.py's saved readers rebuilt (weights, classes, supports)."""
    x = torch.load(EVAL / f"{label}_readers.pt", weights_only=False)["readers"]
    rd = {"g": {}, "p": {}}
    for key, r in x["g"].items():
        name, c = key.split("@")
        sd = r["state_dict"]
        net = nn.Sequential(nn.Linear(sd["0.weight"].shape[1], 128), nn.GELU(), nn.Linear(128, len(r["classes"])))
        net.load_state_dict(sd)
        rd["g"][(name, int(c))] = {"net": net.to(DEV).eval(), "classes": r["classes"], "support": r["support"],
                                   "qualified": r["qualified"]}
    for name, r in x["p"].items():
        rd["p"][name] = {"means": r["means"], "support": r["support"], "qualified": r["qualified"]}
    return rd


def templates(L, cells, n_ep=300):
    """Exact HUD pixel templates per cell and value [values, 147] from dev frames (cells render exactly per value)."""
    eps, s_off = S.episodes(L), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    acc = {c: {} for c in cells}
    for e in range(min(n_ep, len(eps))):
        ob = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"]
        for c, name in cells.items():
            r, k = divmod(c, 9)
            px = ob[:, 7 * r:7 * r + 7, 7 * k:7 * k + 7].reshape(len(ob), -1).float()
            val = EV.hud_value(L, name, np.arange(s_off[e], s_off[e + 1]))
            for v in np.unique(val):
                if int(v) not in acc[c]:
                    acc[c][int(v)] = px[torch.as_tensor(val == v)][0]
    return {c: (np.array(sorted(a)), torch.stack([a[v] for v in sorted(a)])) for c, a in acc.items()}


def cell_px(ob, cell):
    r, k = divmod(cell, 9)
    return ob[7 * r:7 * r + 7, 7 * k:7 * k + 7].reshape(-1).float()


@torch.no_grad()
def decode(model, cfg, g, v, cells):
    """Glimpse tokens [n, 9, 64] -> the fovea reconstruction's decoder mean, uint8 scale [n, 9, 147]."""
    out = []
    for a in range(0, len(g), 1024):
        gg, vv, cc = g[a:a + 1024].float().to(DEV), v[a:a + 1024].float().to(DEV), cells[a:a + 1024].to(DEV)
        with TR.autocast(DEV, cfg):
            mean, _ = model.pixels(model.tsp.reconstruct(gg, vv, cc), vv, cc)
        out.append(((mean.float().clamp(-1, 1) + 1) * 127.5).cpu())
    return torch.cat(out)


def nearest_template(tpl, px):
    vals, T = tpl
    d = torch.cdist(px.float(), T)
    return vals[d.argmin(1).numpy()], d.min(1).values.numpy()


def ratio(rd, name, tok, val, centre, ok):
    """Distance of the cell token to its value's real mean over that value's support radius (nan where not ok)."""
    out = np.full(len(val), np.nan)
    for j in np.nonzero(ok)[0]:
        r = rd["g"][(name, int(centre[j]))]
        s = r["support"].get((0, int(val[j])))
        if s is not None:
            out[j] = float((tok[j].float() - s[0].float()).norm()) / s[1]
    return out


@torch.no_grad()
def field(model, cfg, enc, rows, info, rd, name, gen):
    """Eye-on rows (catalogue rows (e, t, a)): f_up / f_down of the denoiser's glimpse clean estimate per sigma."""
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    keep = [i for i, r in enumerate(rows) if (M.window(enc[r[0]]["fix"][r[1]]) == info[i][0]).any()]
    out = {k: [] for k in ("ctx", "start", "f_up", "f_down", "f_up_real")}
    for b0 in range(0, len(keep), 256):
        idx = keep[b0:b0 + 256]
        take = lambda key, dt: torch.stack([enc[rows[i][0]][key][rows[i][1] + dt] for i in idx]).to(DEV)
        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-P, 0):
                cache = M.context_pass(den, tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v")),
                                       take("prev_action", dt)[:, None], model.belief.memory(take("bp", dt).float())[:, None],
                                       take("fix", dt)[:, None], cache, cfg, gen)
            pa, mem, fx = take("prev_action", 0)[:, None], model.belief.memory(take("bp", 0).float())[:, None], take("fix", 0)[:, None]
            x0 = tuple(take(k, 0).float() for k in ("p", "g", "v"))
            dens = []
            for s in FIELD_SIGMAS:
                xs = tuple(t[:, None] + s * torch.randn(t[:, None].shape, device=DEV, generator=gen) for t in x0)
                Dn, _ = M.denoised(den, xs, torch.full((len(idx), 1), s, device=DEV), pa, mem, fx, cache, clamp, sd)
                dens.append(Dn[1][:, 0].float().cpu())
        fix = fx[:, 0].cpu()
        cell = np.array([info[i][0] for i in idx]); start = np.array([info[i][1] for i in idx]); end = np.array([info[i][2] for i in idx])
        slot, centre = (M.window(fix).numpy() == cell[:, None]).argmax(1), (fix[:, 0] * 9 + fix[:, 1]).numpy()
        real = x0[1].float().cpu()[np.arange(len(idx)), slot]
        fu, fdn, fr = np.full((len(idx), len(FIELD_SIGMAS)), np.nan), np.full((len(idx), len(FIELD_SIGMAS)), np.nan), np.full(len(idx), np.nan)
        for j in range(len(idx)):
            r = rd["g"].get((name, int(centre[j])))
            if r is None or not r["qualified"]:
                continue
            sup = r["support"]
            if (0, int(start[j])) not in sup or (0, int(end[j])) not in sup:
                continue
            ms, me = sup[(0, int(start[j]))][0].float(), sup[(0, int(end[j]))][0].float()
            proj = lambda x, a, b: float(((x - a) * (b - a)).sum() / (b - a).square().sum())
            fr[j] = proj(real[j], ms, me)
            for k in range(len(FIELD_SIGMAS)):
                tok = dens[k][j, slot[j]]
                fu[j, k] = proj(tok, ms, me)
                if (0, int(start[j]) - 1) in sup:
                    fdn[j, k] = proj(tok, ms, sup[(0, int(start[j]) - 1)][0].float())
        for k, v in (("ctx", idx), ("start", start), ("f_up", fu), ("f_down", fdn), ("f_up_real", fr)):
            out[k].append(np.asarray(v))
    return {k: np.concatenate(v) for k, v in out.items()}


def routing_reads(model, cfg, rows, info, enc, rd, kind):
    """evaluate.action_routing's draws (same seeds and batches), keeping the generated value."""
    ctxs = [(i, r[0], r[1]) for i, r in enumerate(rows) if (M.window(enc[r[0]]["fix"][r[1]]) == info[i][0]).any()]
    out = {}
    for variant in ("rerun", "noop_belief"):
        res = {k: [] for k in ("ctx", "eligible", "readable", "read", "start", "end", "success")}
        for b0 in range(0, len(ctxs), 256 // E.R_ROUTING):
            batch = ctxs[b0:b0 + 256 // E.R_ROUTING]
            inps = []
            for i, e, t in batch:
                inp = IP.row_inputs(model, cfg, enc[e], t, {"rerun": True, "noop": variant == "noop_belief"}, None)
                inp["pa"] = inp["pa"].clone(); inp["pa"][-1] = enc[e]["prev_action"][t]
                inp["seed"] = E.ROUTING_SEED + E.ROUTING_KINDS.index(kind) * 100_000 + b0
                inps += [inp] * E.R_ROUTING
            _, gg, _ = E.generate(model, cfg, inps)
            ii = np.repeat([i for i, _, _ in batch], E.R_ROUTING)
            cell = np.array([info[i][0] for i in ii]); start = np.array([info[i][1] for i in ii]); end = np.array([info[i][2] for i in ii])
            fix = torch.stack([enc[rows[i][0]]["fix"][rows[i][1]] for i in ii])
            slot, centre = (M.window(fix).numpy() == cell[:, None]).argmax(1), (fix[:, 0] * 9 + fix[:, 1]).numpy()
            elig = E.eligible_glimpse(rd, E.READS[kind], centre, start, end, np.ones(len(ii), bool))
            read, ok = E.read_glimpse(rd, E.READS[kind], gg, slot, centre, elig)
            for k, v in (("ctx", ii), ("eligible", elig), ("readable", elig & ok), ("read", read), ("start", start),
                         ("end", end), ("success", elig & ok & (read == end))):
                res[k].append(v)
        out[variant] = {k: np.concatenate(v) for k, v in res.items()}
    return out


def outcome_counts(read, start, end, m):
    c = {"n": int(m.sum()), "up": int((m & (read == end)).sum()), "copy": int((m & (read == start) & (read != end)).sum()),
         "down": int((m & (read == start - 1) & (read != end)).sum())}
    c["far"] = c["n"] - c["up"] - c["copy"] - c["down"]
    return c


def summary(x):
    x = x[np.isfinite(x)]
    return None if not len(x) else {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
                                    "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9))}


def draw(path, panels, header):
    """panels: rows of (title, [(px_list of 147-d uint8-scale tensors, caption)]) -> one PNG, cells upscaled x8."""
    sc, w = 8, 7 * 8
    width = 180 + max(len(r[1]) for r in panels) * (4 * w + 24)
    img = Image.new("RGB", (width, 30 + len(panels) * (w + 28)), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.text((4, 4), header, fill=(0, 0, 0))
    for i, (title, items) in enumerate(panels):
        y = 30 + i * (w + 28)
        dr.text((4, y + w // 2), title, fill=(0, 0, 0))
        for j, (pxs, cap) in enumerate(items):
            x = 180 + j * (4 * w + 24)
            for k, px in enumerate(pxs):
                a = px.clamp(0, 255).round().to(torch.uint8).reshape(7, 7, 3).numpy()
                img.paste(Image.fromarray(a).resize((w, w), Image.NEAREST), (x + k * w, y))
            dr.text((x, y + w + 2), cap, fill=(0, 0, 0))
    img.save(path)


def main():
    labels = sys.argv[1:] or list(CKPTS)
    L = dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    W = PM.window_of_state(L, rule)
    eps = S.episodes(L)
    frames = {}
    obs = lambda e: frames.setdefault(e, D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"])
    tpl = templates(L, {65: "drink", 68: "wood", 63: "health"})
    res, rows, info = {"templates": {c: v.tolist() for c, (v, _) in tpl.items()}, "checkpoints": {}}, None, None
    for label in labels:
        model, cfg, _ = E.load(CKPTS[label])
        rd = readers(label)
        enc = EV.encode_with(model, L, cfg, W)
        if rows is None:
            rows, _ = E.rows_of(L, enc, E.N)
            info = {k: EV.row_info(k, rows[k], L) for k in rows}
        hm, tm, means = CAT.hud_means(L, enc), CAT.table_means(L, enc), FC.value_means(L, enc)
        means_all = {**means, 68: hm["wood"], **{c: {0: v} for c, v in tm.items()}}
        R = res["checkpoints"][label] = {"instrument": {}, "field": {}, "routing": {}}
        for kind in KINDS:
            name = E.READS[kind]
            gen = torch.Generator(device=DEV).manual_seed(EV.SEEDS[kind] + 100)
            if kind in EV.CAT_KINDS:
                o = CAT.run(model, enc, L, rows[kind], kind, hm, tm, cfg, gen, heun=False, keep=True)
            else:
                o = FC.run(model, enc, rows[kind], means_all, cfg, gen, keep=True)
            oc = E.outcomes(o, rows[kind], info[kind], enc, rd, kind, "guided", L)
            z = np.load(EVAL / f"{label}_guided_{kind}.npz")
            same = z["eligible"] & oc["eligible"]
            repro = {"eligible_equal": float((z["eligible"] == oc["eligible"]).mean()),
                     "read_equal_on_eligible": float((z["read"][same] == oc["read"][same]).mean()),
                     "readable_equal_on_eligible": float((z["readable"][same] == oc["readable"][same]).mean())}
            idx = o["row_ctx"]
            n = len(idx)
            cell = np.array([info[kind][i][0] for i in idx])
            fix = torch.stack([enc[rows[kind][i][0]]["fix"][rows[kind][i][1]] for i in idx])
            win = M.window(fix)
            slot, centre = (win.numpy() == cell[:, None]).argmax(1), (fix[:, 0] * 9 + fix[:, 1]).numpy()
            gen_g = torch.as_tensor(o["gen_g"]).float()
            real_g = torch.stack([enc[rows[kind][i][0]]["g"][rows[kind][i][1]] for i in idx]).float()
            real_v = torch.stack([enc[rows[kind][i][0]]["v"][rows[kind][i][1]] for i in idx]).float()
            el, ok = oc["eligible"], oc["readable"]
            r_gen = ratio(rd, name, gen_g[np.arange(n), slot], oc["read"], centre, ok)
            r_end = ratio(rd, name, gen_g[np.arange(n), slot], oc["end"], centre, ok)
            r_start = ratio(rd, name, gen_g[np.arange(n), slot], oc["start"], centre, ok)
            r_real = ratio(rd, name, real_g[np.arange(n), slot], oc["observed"], centre, el)
            dec_gen = decode(model, cfg, gen_g, torch.as_tensor(o["gen_v"]).float(), win)[np.arange(n), slot]
            dec_real = decode(model, cfg, real_g, real_v, win)[np.arange(n), slot]
            c0 = HUD_CELL[name]
            pv_gen, pd_gen = nearest_template(tpl[c0], dec_gen)
            pv_real, pd_real = nearest_template(tpl[c0], dec_real)
            st, en, rv = oc["start"], oc["end"], oc["read"]
            cat = np.where(rv == en, "success", np.where(rv == st, "copy", np.where(rv == st - 1, "down", "far")))
            groups = {"appearance": st == 0, "increment": st >= 1} if kind in ("drink", "wood") else {"all": st >= 0}
            R["instrument"][kind] = {"reproduction": repro, "groups": {}}
            for gname, gm in groups.items():
                gi = {"real_token_ratio": summary(r_real[el & gm]),
                      "pixel_real_agrees_observed": float((pv_real[el & gm] == oc["observed"][el & gm]).mean()) if (el & gm).any() else None,
                      "pixel_real_template_distance": summary(pd_real[el & gm]), "by_category": {}}
                for c in ("success", "copy", "down", "far"):
                    m = ok & gm & (cat == c)
                    gi["by_category"][c] = {"n": int(m.sum()), "token_ratio": summary(r_gen[m]),
                                            "inside_end_support": float((r_end[m] <= 1).mean()) if m.any() else None,
                                            "inside_start_support": float((r_start[m] <= 1).mean()) if m.any() else None,
                                            "pixel_agrees_token_read": float((pv_gen[m] == rv[m]).mean()) if m.any() else None,
                                            "pixel_template_distance": summary(pd_gen[m]),
                                            "pixel_values": {int(a): int(b) for a, b in zip(*np.unique(pv_gen[m], return_counts=True))}}
                m = el & ~ok & gm
                gi["unknown"] = {"n": int(m.sum()), "pixel_values": {int(a): int(b) for a, b in zip(*np.unique(pv_gen[m], return_counts=True))},
                                 "pixel_template_distance": summary(pd_gen[m])}
                R["instrument"][kind]["groups"][gname] = gi
                if label in ("base", "guided_0750", "guided_1500") and kind in ("drink", "drink_fail", "wood"):
                    panels = []
                    for c in ("success", "copy", "down", "far"):
                        js = np.nonzero(ok & gm & (cat == c))[0][:IMG_PER_CAT]
                        items = []
                        for j in js:
                            e, t = rows[kind][idx[j]][0], rows[kind][idx[j]][1]
                            ob = obs(e)
                            items.append(([cell_px(ob[t - 1], cell[j]), cell_px(ob[t], cell[j]), dec_real[j], dec_gen[j]],
                                          f"{st[j]}->{en[j]} read {rv[j]} px {pv_gen[j]}"))
                        if items:
                            panels.append((f"{c} ({int((ok & gm & (cat == c)).sum())})", items))
                    if panels:
                        draw(HERE / f"img_{label}_{kind}_{gname}.png", panels,
                             f"{label} {kind} {gname}: real t-1 | real t | decoded real t | decoded generated t")
        for kind in FIELD_KINDS:
            fo = field(model, cfg, enc, rows[kind], info[kind], rd, E.READS[kind], torch.Generator(device=DEV).manual_seed(7))
            R["field"][kind] = {}
            for gname, gm in {"appearance": fo["start"] == 0, "increment": fo["start"] >= 1}.items():
                R["field"][kind][gname] = {"f_up_real": summary(fo["f_up_real"][gm]),
                                           "f_up": {str(s): summary(fo["f_up"][gm, k]) for k, s in enumerate(FIELD_SIGMAS)},
                                           "f_down": {str(s): summary(fo["f_down"][gm, k]) for k, s in enumerate(FIELD_SIGMAS)}}
        for kind in ("drink", "wood"):
            ar = routing_reads(model, cfg, rows[kind], info[kind], enc, rd, kind)
            z = np.load(EVAL / f"{label}_routing_{kind}.npz")
            R["routing"][kind] = {"reproduction": {v: float((z[f"{v}_success"] == ar[v]["success"]).mean()) for v in ar}}
            for v, a in ar.items():
                for gname, gm in {"appearance": a["start"] == 0, "increment": a["start"] >= 1}.items():
                    R["routing"][kind].setdefault(gname, {})[v] = outcome_counts(a["read"], a["start"], a["end"], a["readable"] & gm)
        print(label, json.dumps({k: R["instrument"][k]["reproduction"] for k in KINDS}), flush=True)
        (HERE / "glimpse_field.json").write_text(json.dumps(res, indent=1, default=str))
        del model, enc
        torch.cuda.empty_cache()
    print("saved", HERE / "glimpse_field.json", flush=True)


if __name__ == "__main__":
    main()
