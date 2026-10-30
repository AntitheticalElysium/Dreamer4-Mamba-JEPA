"""Part 2 of PLAN.md: the 10-29 loss and gradient accounting (generation_diagnosis/loss_accounting.py: its method,
train-400 windows, noise and Adam drive) with drink and wood split into appearance (start 0) and increment (start >= 1)
and the glimpse elements of every glimpse token on a consequence cell attributed to that consequence (g_*).
Fixations: 'random' (the S1-seeded prior, as P1 and the 10-29 run) or 'guided' (every attempt frame fixated by the
placement rule, Q = 1; the fine-tune's Q = 1/2 is their midpoint in expectation: loss and gradient sums are linear in
the mixture). Split halves as the 10-29 run (alternating windows), for noise-free norms, cosines and drives.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_increment_diagnosis.signal_accounting CKPT FIX OUT_JSON
  [WINDOWS]
"""
import importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
PM = EV.PM
ROOT, DEV, P, T, B = LA.ROOT, LA.DEV, LA.P, LA.T, LA.B
CONSEQ = ("drink_appear", "drink_incr", "wood_appear", "wood_incr", "sapling_up", "table_placed", "zombie_hit")
CLASSES = CONSEQ + ("health_other", "hud_other_change", "hud_static", "map_moved", "map_still",
                    "g_drink_appear", "g_drink_incr", "g_wood_appear", "g_wood_incr", "g_other_consequence", "g_rest",
                    "lighting")
PERIPHERY = CLASSES[:12]


def classes(L, enc):
    """Per state: periphery class [81] and per glimpse token class [9] (indices into CLASSES)."""
    old = LA.cell_classes(L)
    n = len(old)
    prev = np.maximum(np.arange(n) - 1, 0)
    cls = np.zeros_like(old)
    for i, name in enumerate(LA.CLASSES):
        if name in PERIPHERY:
            cls[old == i] = CLASSES.index(name)
    for name, cell, start in (("drink", 65, L["s_body"][prev, 2]), ("wood", 68, L["s_inv"][prev, 0])):
        up = old[:, cell] == LA.CLASSES.index(f"{name}_up")
        cls[up, cell] = np.where(start[up] == 0, CLASSES.index(f"{name}_appear"), CLASSES.index(f"{name}_incr"))
    win = M.window(torch.cat([e["fix"] for e in enc])).numpy()
    under = np.take_along_axis(cls, win, 1)
    g = np.full(win.shape, CLASSES.index("g_rest"), np.int8)
    for name in CONSEQ:
        g[under == CLASSES.index(name)] = CLASSES.index("g_" + name if name[:4] in ("drin", "wood") else "g_other_consequence")
    g[np.r_[0, np.cumsum(L["ep_T"] + 1)][:-1]] = CLASSES.index("g_rest")  # first frames: cell_classes compares across episodes
    return cls, g


def main():
    ckpt, fix, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); out.parent.mkdir(parents=True, exist_ok=True)
    n_win = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz"))
    if fix == "random":
        enc = S.encode(model, L, cfg)
    else:
        rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
        enc = EV.encode_with(model, L, cfg, PM.window_of_state(L, rule))
    cls_cell, cls_g = classes(L, enc)
    wins, s_off = LA.windows(L, enc, n_win, np.random.default_rng(2033))
    den = model.denoiser
    params = [p for p in den.parameters() if p.requires_grad]
    GH = {c: [[torch.zeros(p.shape) for p in params] for _ in range(2)] for c in CLASSES}
    nC = len(CLASSES)
    loss = np.zeros(nC); elems = np.zeros(nC); events = np.zeros(nC); frames = 0
    gen = torch.Generator(device=DEV).manual_seed(2033)
    den.train()
    for start in range(0, len(wins), B):
        batch = wins[start:start + B]
        take = lambda key: torch.stack([enc[e][key][s:s + P + T] for e, s in batch]).to(DEV)
        x0 = tuple(take(k).float() for k in ("p", "g", "v"))
        pa, fx = take("prev_action"), take("fix")
        mem = model.belief.memory(take("bp").float()).detach()
        first = torch.stack([torch.as_tensor(np.arange(s, s + P + T) == 0) for e, s in batch]).to(DEV)
        pa = torch.where(first, torch.full_like(pa, -1), pa)
        episode = torch.tensor([[i] * (P + T) for i in range(len(batch))], device=DEV)
        sigma = TR.sample_sigma(episode.shape, cfg, DEV, gen)
        noise = tuple(torch.randn(t.shape, device=DEV, generator=gen) for t in x0)
        with TR.autocast(DEV, cfg):
            c_skip, c_out, c_in, c_noise = M.precondition(sigma, cfg["edm"]["sigma_data"])
            xs = tuple(t + M._bcast(sigma, t) * n for t, n in zip(x0, noise))
            Fo = den(tuple(M._bcast(c_in, t) * t for t in xs), c_noise, pa, mem, fx, episode, torch.ones_like(episode, dtype=torch.bool))
        se = [((f.float() - (t - M._bcast(c_skip, t) * s) / M._bcast(c_out, t)) ** 2)[:, P:] for f, t, s in zip(Fo, x0, xs)]
        sl = lambda a: torch.stack([torch.as_tensor(a[s_off[e] + s + P:s_off[e] + s + P + T].astype(np.int64)) for e, s in batch]).to(DEV)
        cc, cg = sl(cls_cell), sl(cls_g)
        keep = ~first[:, P:]
        sp, sg, sv = se[0].sum(-1), se[1].sum(-1), se[2].sum(-1)          # [b, T, 81], [b, T, 9], [b, T]
        norm = 1880.0 * T * len(batch)
        per = []
        for k, name in enumerate(CLASSES):
            if name == "lighting":
                v, cnt, ev = sv, torch.full_like(sv, 8), None
            elif name.startswith("g_"):
                m = cg == k
                v, cnt, ev = (sg * m).sum(-1), m.sum(-1).float() * 64, m.any(-1)
            else:
                m = (cc == k) & keep[..., None]
                v, cnt, ev = (sp * m).sum(-1), m.sum(-1).float() * 16, m.any(-1)
            per.append(v.sum() / norm)
            loss[k] += float(v.detach().sum()); elems[k] += float(cnt.sum())
            events[k] += 0 if ev is None else int(ev.sum())
        frames += T * len(batch)
        half = (start // B) % 2
        for k, name in enumerate(CLASSES):
            g = torch.autograd.grad(per[k], params, retain_graph=k < nC - 1, allow_unused=True)
            for acc, gi in zip(GH[name][half], g):
                if gi is not None:
                    acc += gi.float().cpu() * len(batch)
        if start % (B * 50) == 0:
            print(f"window {start}/{len(wins)}", flush=True)
    n_half = [sum(len(wins[s0:s0 + B]) for s0 in range(h * B, len(wins), 2 * B)) for h in range(2)]
    GH = {c: [[g / n_half[h] for g in GH[c][h]] for h in range(2)] for c in CLASSES}
    w_g1 = 1.0 / float(ck["model"]["harmoniser.log_sigma"][0].exp())
    logs = [json.loads(l) for l in (ckpt.parent / "log.jsonl").read_text().splitlines()]
    done = [r for r in logs if "grad_norm" in r and r.get("update", 0) <= ck["state"]["update"]][-40:]
    clip = float(np.mean([min(1.0, o["grad_clip"] / max(r["grad_norm"], 1e-12)) for r in done]))
    b1, b2 = o["betas"]
    group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
    pre, lr_of = [], []
    for p in params:
        st = opt.state[p]
        vhat = st["exp_avg_sq"].float().cpu() / (1 - b2 ** float(st["step"]))
        pre.append(1.0 / (vhat.sqrt() + o["eps"])); lr_of.append(group_of[id(p)]["lr"])
    dot = lambda a, b: float(sum((x * y).sum() for x, y in zip(a, b)))
    step = lambda Gv: [-lr_ * clip * w_g1 * g * q for g, q, lr_ in zip(Gv, pre, lr_of)]
    GtH = [[sum(GH[c][h][i] for c in CLASSES) for i in range(len(params))] for h in range(2)]
    dA, dB = step(GtH[0]), step(GtH[1])
    tt = dot(GtH[0], GtH[1])
    res = {"checkpoint": str(ckpt), "fixations": fix, "update": ck["state"]["update"], "windows": len(wins),
           "target_frames": frames, "g1_weight": w_g1, "mean_clip": clip, "lr": [g["lr"] for g in opt.param_groups],
           "total_loss": float(loss.sum() / (1880.0 * frames)), "classes": {}}
    for k, name in enumerate(CLASSES):
        A_, B_ = GH[name]
        cc_ = dot(A_, B_)
        Lc = float(loss[k] / (1880.0 * frames))
        res["classes"][name] = {
            "events": int(events[k]), "elements": float(elems[k]), "loss": Lc, "loss_share": float(loss[k] / loss.sum()),
            "se_per_element": float(loss[k] / max(elems[k], 1)),
            "reliability_cos_A_B": cc_ / (math.sqrt(dot(A_, A_) * dot(B_, B_)) + 1e-30),
            "grad_norm_share": math.sqrt(max(cc_, 0.0) / max(tt, 1e-30)),
            "cos_with_total": (0.5 * (dot(A_, GtH[1]) + dot(B_, GtH[0])) / math.sqrt(cc_ * tt)) if cc_ > 0 and tt > 0 else None,
            "drive_per_1000": 1000 * 0.5 * (dot(A_, dB) + dot(B_, dA)) / max(Lc, 1e-30),
            "own_drive_per_1000": 1000 * dot(A_, step(B_)) / max(Lc, 1e-30)}
        r = res["classes"][name]
        print(name, {a: (round(b, 5) if isinstance(b, float) else b) for a, b in r.items()}, flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
