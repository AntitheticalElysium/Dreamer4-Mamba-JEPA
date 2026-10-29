"""Bounded accounting of P1's G1 training signal by consequence class (user, Codex; 10-29). One GPU pass per checkpoint.

QUESTION: how much of the weighted denoising objective, and of its gradient as Adam applies it, concerns each
consequence (drink, wood, sapling, table, zombie hit) beside movement and the static bulk; does the update direction at
the checkpoint reduce each class's loss, and how fast relative to its size?
COMPETING READINGS: (a) single-cell consequences carry a tiny share of loss and gradient and the update barely reduces
their loss: a loss-allocation account is plausible; (b) their share or their rate of decrease is comparable to movement:
allocation is not the limit (optimisation, capacity, interference remain); (c) the update increases their loss:
interference with the bulk.
METHOD, the exact P1 G1 objective (train.p1_losses -> model.denoise_loss): train-400 episodes; perception and belief
over whole episodes with the S1 seeded random fixations (the TBTT-carried belief); the denoiser in its chunked training
form over windows of 8 prefix + 64 frames of one episode, diffusion forcing (each frame's sigma ~ p_train, prefix frames
included), EDM preconditioning, raw-output MSE; G1 = sum of squared raw-output errors / (1,880 scalars x 64 frames).
Every element of every target frame belongs to exactly one class (labels of frame t vs t - 1), so the class losses sum
to G1. Gradients of each class's loss w.r.t. the denoiser parameters (G1 is the only loss reaching them; the
harmoniser weight 1 / sigma_G1 and clipping scale every class alike). The expected update uses the checkpoint's own
Adam state: dtheta = -lr * c * E[g] / (sqrt(v_hat) + eps) with E[g] = (1 / sigma_G1) * the accumulated G1 gradient and
c the mean clip factor of the logged final gradient norms; first-order change of class c's loss per step = G_c .
dtheta; also under the stored momentum and under AdamW's decay. Loss by equal-mass p_train sigma bins.
Split halves (added 10-29 after the first runs): gradients are accumulated separately on alternating windows (A, B);
for rare classes sampling noise inflates |G_c| and the own-gradient term and shrinks cosines, so the cross-half
products G_c^A . G_c^B (unbiased for |G_c|^2) and G_c^A . dtheta(G^B) are the reported "_cross" quantities.
Not established by this: that allocation causes the deficit (plausible, not proof); per-batch noise is ignored.
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.loss_accounting CKPT OUT_JSON [WINDOWS]
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

ROOT = Path(__file__).resolve().parents[3]
DEV = torch.device("cuda")
P, T, B = 8, 64, 4
HUD_SRC = {63: ("s_body", 0), 64: ("s_body", 1), 65: ("s_body", 2), 66: ("s_body", 3), 67: ("s_inv", 5),
           68: ("s_inv", 0), 69: ("s_inv", 1), 70: ("s_inv", 2), 71: ("s_inv", 3), 72: ("s_inv", 4),
           73: ("s_inv", 6), 74: ("s_inv", 7), 75: ("s_inv", 8), 76: ("s_inv", 9), 77: ("s_inv", 10), 78: ("s_inv", 11)}
CLASSES = ("drink_up", "wood_up", "sapling_up", "table_placed", "zombie_hit", "health_other", "hud_other_change",
           "hud_static", "map_moved", "map_still", "glimpse", "lighting")
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
TABLE = 11


def cell_classes(L):
    """Per state t (t >= 1 within its episode): class index of each of the 81 periphery cells of frame t."""
    n = len(L["s_body"])
    cls = np.full((n, 81), CLASSES.index("map_still"), np.int8)
    prev = np.maximum(np.arange(n) - 1, 0)
    moved = (L["s_pos"] != L["s_pos"][prev]).any(-1)
    cls[moved, :63] = CLASSES.index("map_moved")
    d = DIRS[L["s_dir"][prev]]
    fr, fc = 3 + d[:, 0], 4 + d[:, 1]
    placed = (L["s_view"][np.arange(n), fr, fc] == TABLE) & (L["s_view"][prev, fr, fc] != TABLE) & ~moved
    cls[placed, fr[placed] * 9 + fc[placed]] = CLASSES.index("table_placed")
    hit = (L["s_zombies_m"][prev] & (L["s_zombies_cd"][prev] <= 0) & (L["s_zombies_cd"] == 5)).any(1)
    for cell in range(63, 81):
        if cell not in HUD_SRC:
            cls[:, cell] = CLASSES.index("hud_static")
            continue
        a, col = HUD_SRC[cell]
        now, before = L[a][:, col], L[a][prev, col]
        c = np.where(now != before, CLASSES.index("hud_other_change"), CLASSES.index("hud_static"))
        if cell == 63:
            c = np.where(hit, CLASSES.index("zombie_hit"), np.where(now != before, CLASSES.index("health_other"), c))
        for name, cc in (("drink_up", 65), ("wood_up", 68), ("sapling_up", 67)):
            if cell == cc:
                c = np.where(now == before + 1, CLASSES.index(name), c)
        cls[:, cell] = c
    return cls


def windows(L, enc, n, rng):
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    cand = [(e, s) for e in range(len(enc)) for s in range(0, len(enc[e]["p"]) - (P + T) + 1, T)]
    pick = rng.choice(len(cand), min(n, len(cand)), replace=False)
    return [cand[i] for i in pick], s_off


def main():
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]); out.parent.mkdir(parents=True, exist_ok=True)
    n_win = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz"))
    enc = S.encode(model, L, cfg)
    cls_cell = cell_classes(L)
    wins, s_off = windows(L, enc, n_win, np.random.default_rng(2033))
    den = model.denoiser
    params = [p for p in den.parameters() if p.requires_grad]
    names = [n for n, p in den.named_parameters() if p.requires_grad]
    G = {c: [torch.zeros_like(p, dtype=torch.float32) for p in params] for c in CLASSES}
    GH = {c: [[torch.zeros_like(p, dtype=torch.float32) for p in params] for _ in range(2)] for c in CLASSES}
    nC = len(CLASSES)
    edges = torch.distributions.Normal(0, 1).icdf(torch.arange(1, 8) / 8.0)
    s_cfg = cfg["edm"]["train_sigma"]
    sig_edges = (s_cfg["lognormal_loc"] + s_cfg["lognormal_scale"] * edges).exp().to(DEV)
    loss_bin = torch.zeros(nC, 8, dtype=torch.float64); elems_bin = torch.zeros(nC, 8, dtype=torch.float64)
    events = np.zeros(nC); frames = 0
    gen = torch.Generator(device=DEV).manual_seed(2033)
    den.train()                                                          # no dropout exists; checkpointing stays off
    for start in range(0, len(wins), B):
        batch = wins[start:start + B]
        take = lambda key: torch.stack([enc[e][key][s:s + P + T] for e, s in batch]).to(DEV)
        x0 = tuple(take(k).float() for k in ("p", "g", "v"))
        pa, fix = take("prev_action"), take("fix")
        mem = model.belief.memory(take("bp").float()).detach()
        first = torch.stack([torch.as_tensor(np.arange(s, s + P + T) == 0) for e, s in batch]).to(DEV)
        pa = torch.where(first, torch.full_like(pa, -1), pa)
        episode = torch.tensor([[i] * (P + T) for i in range(len(batch))], device=DEV)
        valid = torch.ones_like(episode, dtype=torch.bool)
        sigma = TR.sample_sigma(episode.shape, cfg, DEV, gen)
        noise = tuple(torch.randn(t.shape, device=DEV, generator=gen) for t in x0)
        with TR.autocast(DEV, cfg):
            c_skip, c_out, c_in, c_noise = M.precondition(sigma, cfg["edm"]["sigma_data"])
            xs = tuple(t + M._bcast(sigma, t) * n for t, n in zip(x0, noise))
            Fo = den(tuple(M._bcast(c_in, t) * t for t in xs), c_noise, pa, mem, fix, episode, valid)
        se = [((f.float() - (t - M._bcast(c_skip, t) * s) / M._bcast(c_out, t)) ** 2)[:, P:] for f, t, s in zip(Fo, x0, xs)]
        # class index per element: periphery [b, T, 81] by cell, glimpse and lighting their own
        cc = torch.stack([torch.as_tensor(cls_cell[s_off[e] + s + P:s_off[e] + s + P + T].astype(np.int64)) for e, s in batch]).to(DEV)
        first_t = first[:, P:]
        norm = 1880.0 * T * len(batch)
        per_cls = []
        sb = torch.bucketize(sigma[:, P:], sig_edges)                    # [b, T] equal-mass bin
        sp = se[0].sum(-1)                                               # [b, T, 81]
        for k, name in enumerate(CLASSES):
            if name == "glimpse":
                v = se[1].sum((-1, -2))
                cnt = torch.full_like(v, 9 * 64)
            elif name == "lighting":
                v = se[2].sum(-1)
                cnt = torch.full_like(v, 8)
            else:
                m = (cc == k) & ~first_t[..., None]
                v = (sp * m).sum(-1)
                cnt = m.sum(-1).float() * 16
                events[k] += int((m.any(-1)).sum()) if k < 6 else 0
            per_cls.append(v.sum() / norm)
            for b_ in range(8):
                sel = sb == b_
                loss_bin[k, b_] += float(v[sel].detach().sum()); elems_bin[k, b_] += float(cnt[sel].sum())
        frames += int(T * len(batch))
        for k, name in enumerate(CLASSES):
            g = torch.autograd.grad(per_cls[k], params, retain_graph=k < nC - 1, allow_unused=True)
            half = (start // B) % 2
            for acc, acc_h, gi in zip(G[name], GH[name][half], g):
                if gi is not None:
                    acc += gi.float() * len(batch)
                    acc_h += gi.float() * len(batch)
        if start % (B * 50) == 0:
            print(f"window {start}/{len(wins)}", flush=True)
    nb = len(wins)
    n_half = [sum(len(wins[s0:s0 + B]) for s0 in range(h * B, len(wins), 2 * B)) for h in range(2)]
    for name in CLASSES:
        G[name] = [g / nb for g in G[name]]                              # gradient of the per-window mean G1
        GH[name] = [[g / n_half[h] for g in GH[name][h]] for h in range(2)]
    Gt = [sum(G[c][i] for c in CLASSES) for i in range(len(params))]
    # Adam state for the denoiser parameters
    w_g1 = 1.0 / float(ck["model"]["harmoniser.log_sigma"][0].exp())
    logs = [json.loads(l) for l in (ckpt.parent / "log.jsonl").read_text().splitlines()]
    done = [r for r in logs if "grad_norm" in r and r.get("update", 0) <= ck["state"]["update"]][-40:]
    clip = float(np.mean([min(1.0, cfg["optimizer"]["grad_clip"] / max(r["grad_norm"], 1e-12)) for r in done]))
    b1, b2 = cfg["optimizer"]["betas"]
    pre, lr_of, dth_e, dth_m, dth_d = [], [], [], [], []
    group_of = {id(p): g for g in opt.param_groups for p in g["params"]}
    for p, gt in zip(params, Gt):
        st, grp = opt.state[p], group_of[id(p)]
        step = float(st["step"])
        vhat = st["exp_avg_sq"].float() / (1 - b2 ** step)
        mhat = st["exp_avg"].float() / (1 - b1 ** step)
        q = 1.0 / (vhat.sqrt() + cfg["optimizer"]["eps"])
        lr = grp["lr"]
        pre.append(q); lr_of.append(lr)
        dth_e.append(-lr * clip * w_g1 * gt * q); dth_m.append(-lr * mhat * q); dth_d.append(-lr * grp["weight_decay"] * p.detach().float())
    res = {"checkpoint": str(ckpt), "update": ck["state"]["update"], "windows": nb, "target_frames": frames,
           "harmoniser_sigma": ck["model"]["harmoniser.log_sigma"].exp().tolist(), "g1_weight": w_g1, "mean_clip": clip,
           "lr": [g["lr"] for g in opt.param_groups], "classes": {}}
    tot_loss = sum(loss_bin[k].sum() for k in range(nC))
    dot = lambda a, b: float(sum((x * y).sum() for x, y in zip(a, b)))
    gnorm = lambda a: math.sqrt(dot(a, a))
    gt_norm = gnorm(Gt)
    for k, name in enumerate(CLASSES):
        Lc = float(loss_bin[k].sum() / (1880.0 * frames))
        res["classes"][name] = {
            "events": int(events[k]) if k < 6 else None, "elements": float(elems_bin[k].sum()),
            "loss": Lc, "loss_share": float(loss_bin[k].sum() / tot_loss),
            "mean_se_per_element": float(loss_bin[k].sum() / max(elems_bin[k].sum(), 1)),
            "mean_se_per_element_by_sigma_bin": (loss_bin[k] / elems_bin[k].clamp(min=1)).tolist(),
            "loss_share_by_sigma_bin": (loss_bin[k] / loss_bin.sum(0)).tolist(),
            "grad_norm": gnorm(G[name]), "grad_norm_share": gnorm(G[name]) / gt_norm,
            "cos_with_total": dot(G[name], Gt) / (gnorm(G[name]) * gt_norm + 1e-30),
            "dL_per_step_expected_update": dot(G[name], dth_e), "dL_per_step_momentum": dot(G[name], dth_m),
            "dL_per_step_weight_decay": dot(G[name], dth_d),
            "relative_dL_per_1000_steps_expected": 1000 * dot(G[name], dth_e) / max(Lc, 1e-30),
            "relative_dL_per_1000_steps_momentum": 1000 * dot(G[name], dth_m) / max(Lc, 1e-30),
            "relative_dL_per_1000_steps_own_gradient_only":
                1000 * dot(G[name], [-lr_ * clip * w_g1 * g * q for g, q, lr_ in zip(G[name], pre, lr_of)]) / max(Lc, 1e-30)}
        # where the class's gradient lives
        blocks = {}
        for nm, g in zip(names, G[name]):
            key = nm.split(".")[0] if not nm.startswith("blocks.") else "blocks"
            blocks[key] = blocks.get(key, 0.0) + float(g.square().sum())
        res["classes"][name]["grad_sq_by_block"] = blocks
        r = res["classes"][name]
        print(name, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (list, dict))}, flush=True)
    GtH = [[sum(GH[c][h][i] for c in CLASSES) for i in range(len(params))] for h in range(2)]
    step = lambda Gv: [-lr_ * clip * w_g1 * g * q for g, q, lr_ in zip(Gv, pre, lr_of)]
    dA, dB = step(GtH[0]), step(GtH[1])
    tt = dot(GtH[0], GtH[1])
    for name in CLASSES:
        A_, B_ = GH[name]
        cc_ = dot(A_, B_)
        Lc = res["classes"][name]["loss"]
        res["classes"][name]["cross"] = {
            "reliability_cos_A_B": cc_ / (gnorm(A_) * gnorm(B_) + 1e-30),
            "grad_norm_corrected": math.sqrt(max(cc_, 0.0)), "grad_norm_share_corrected": math.sqrt(max(cc_, 0.0) / max(tt, 1e-30)),
            "cos_with_total_corrected": 0.5 * (dot(A_, GtH[1]) + dot(B_, GtH[0])) / (math.sqrt(max(cc_, 1e-30)) * math.sqrt(max(tt, 1e-30))),
            "relative_dL_per_1000_steps_expected": 1000 * 0.5 * (dot(A_, dB) + dot(B_, dA)) / max(Lc, 1e-30),
            "relative_dL_per_1000_steps_expected_halves": [1000 * dot(A_, dB) / max(Lc, 1e-30), 1000 * dot(B_, dA) / max(Lc, 1e-30)],
            "relative_dL_per_1000_steps_own_gradient_only": 1000 * dot(A_, step(B_)) / max(Lc, 1e-30)}
        print(name, "cross", {k: (round(v, 4) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in res["classes"][name]["cross"].items()}, flush=True)
    res["sigma_bin_edges"] = sig_edges.tolist()
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
