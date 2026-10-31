"""Learned-fixation premise test B (PLAN.md B; PROGRESS.md section 7): what should a fixation chooser's progress be OF?
Reads the ten checkpoints of P1 world 1 and of P1-F; no weight changes.
B1 (token drift): per checkpoint, the G1 loss by class (M0.5's class_losses), each class's own frame-to-frame token
change (copy-previous error, clean token units) and, from the previous checkpoint, the per-token least-squares map's
residual (fit on even episodes, read on odd), on the first 150 dev episodes under S1's seeded fixations.
B2 (transfer, first order): the gradient of the exact training objective (train.p1_losses, harmonised + noise) on
chunks of train episodes (8 prefix frames, the attempt frame t, 7 frames after), with only frame t's fixation varied
(counter window, faced-cell window, control window (1, 1), uniform draws), in three geometries (euclid; adam_linear =
lr / (sqrt(v_hat) + eps); adam_step = LESS's Gamma, the full Adam update of a step on this chunk), against the
gradients of gaze-free targets on fresh dev frames: the counter cell's and faced cell's denoising loss and the reward
readout's NLL (query and history fixations uniform; tokens stop-gradient as in training), the counter loss with the
tokens moving with the encoder (counter_full), and the bulk periphery loss. Per module, per episode-disjoint group
(Delta) and half (targets). audit_b.py checks it by finite differences.
Usage (project venv): python -m artifacts.experiments.20261031_learned_fixation_sweep.progress OUT_DIR [--smoke]
(GPU; with CUDA hidden it runs on CPU with Mamba's reference recurrence, for the smoke check);
python -m ...progress --summary OUT_DIR (CPU: the PLAN readings from the saved arrays, summary.json).
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
EXP = Path(__file__).resolve().parents[1]
WORLDS = {"w1": (TR.ROOT / "artifacts/saccade_m0/p1_world1", False), "p1f": (EXP / "20261030_p1_guided/run", True)}
CLASSES = {"drink": (3, 65, None), "wood": (5, 68, None), "sapling": (2, 67, None), "stone": (4, 69, 6)}  # faced block, counter cell, tool
CONDS = ("counter", "faced", "control")
KINDS = ("counter", "faced", "reward", "counter_full")
MODULES = ("perception", "belief", "denoiser", "pixels", "tsp", "change", "outcome")
GEOMETRIES = ("euclid", "adam_linear", "adam_step")
CHANGING = ("drink_up", "wood_up", "sapling_up", "table_placed", "zombie_hit", "health_other", "hud_other_change",
            "map_moved", "lighting")                                    # classes whose target is a change (B1 test)
P, H = 8, 8                                                              # prefix frames; loss frames from t
ROWS = 8                                                                 # rows per backward (6 GB card)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classes_at(L, c, t_lo, after, keep):
    """(episode, t): the class context at state t - 1 (awake, DO, facing the block, counter below 9, tool held)."""
    block, cell, tool = CLASSES[c]
    k, col = LA.HUD_SRC[cell]
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    out = []
    for e in np.flatnonzero(keep):
        t = np.arange(t_lo, L["ep_T"][e] - after + 1)
        g = s_off[e] + t - 1
        m = (L["t_action"][a_off[e] + t - 1] == FC.DO) & ~L["s_sleep"][g].astype(bool) & (face[g] == block) & (L[k][g, col] < 9)
        if tool is not None:
            m &= L["s_inv"][g, tool] > 0
        out += [(int(e), int(x)) for x in t[m]]
    return out


def pick(rows, part, n, parts, rng, cap=4):
    """n rows per part (part[e] in range(parts)), at most `cap` per episode, in a seeded order."""
    out, used = [[] for _ in range(parts)], {}
    for i in rng.permutation(len(rows)):
        e, t = rows[i]
        if len(out[part[e]]) < n and used.get(e, 0) < cap:
            out[part[e]].append((e, t)); used[e] = used.get(e, 0) + 1
    return out


def gaze(L, cfg, windows=None):
    """Per episode: S1's seeded fixations (saccade.eval.s1.encode), the rule's window on rule-decided frames if given."""
    s_off, out = np.r_[0, np.cumsum(L["ep_T"] + 1)], []
    for e, ep in enumerate(S.episodes(L)):
        rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
        fx = np.array([D.fixation_prior(rng, ()) for _ in range(ep.T + 1)])
        if windows is not None:
            w = windows[s_off[e]:s_off[e + 1] - 1]
            fx[1:][w[:, 0] >= 0] = w[w[:, 0] >= 0]
        out.append(fx)
    return out


def load(episodes, index, fx, t_last, offsets=None):
    """Chunk arrays of frames 0..t_last of one episode (saccade.data.gather), as one row."""
    return {k: v[0] for k, v in D.gather(episodes, [[(index, 1, t, *fx[t]) for t in range(t_last + 1)]], offsets).items()}


@torch.no_grad()
def prepass(model, cfg, data, cuts):
    """Per episode: the Mamba carry before each cut frame and every frame's predicted belief (training's carried state)."""
    out = {}
    for key, arr in data.items():
        c = TR.tensors({k: v[None] for k, v in arr.items()}, DEV)
        n = c["first"].shape[1]
        with TR.autocast(DEV, cfg):
            Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                            for a in range(0, n, 512)])), c["fixation"])
            tok = model.belief.tokens(model.belief.percept_embed(Pe), c["reward"], c["terminated"], c["action"])
            carry, at, ys, carries = model.belief.initial(1, DEV), 0, [], {}
            for cut in sorted(cuts[key]) + [n]:
                y, carry = model.belief(tok[:, at:2 * cut], carry)
                ys.append(y); at = 2 * cut
                carries[cut] = tuple(M.MambaCarry(q.conv.cpu(), q.ssm.cpu()) for q in carry)
        y = torch.cat(ys, 1)
        out[key] = {"carry": carries, "bp": torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)[0].float().cpu()}
    return out


def batch(data, pre, rows, a, b, cut):
    """Chunk rows [t + a, t + b) of each (episode, t), with the carry before frame t + cut and the cache bp[t + a, t + cut]."""
    chunk = {k: np.stack([data[e][k][t + a:t + b] for e, t in rows]) for k in data[rows[0][0]]}
    carry = tuple(M.MambaCarry(torch.cat([pre[e]["carry"][t + cut][l].conv for e, t in rows]).to(DEV),
                               torch.cat([pre[e]["carry"][t + cut][l].ssm for e, t in rows]).to(DEV))
                  for l in range(len(pre[rows[0][0]]["carry"][rows[0][1] + cut])))
    cache = torch.stack([pre[e]["bp"][t + a:t + cut + 1] for e, t in rows]).to(DEV)
    return chunk, carry, cache


def flat(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).float().reshape(-1) for g, p in zip(grads, params)])


def target_terms(model, cfg, chunk, carry, cache, cells, gen, full=False):
    """Gaze-free prediction of each row's last frame: denoising error on its counter and faced cells (training units),
    the reward readout's NLL and the bulk periphery error; percepts and belief of the 7 frames before carry gradient.
    Targets and context tokens are stop-gradient as in training (prediction at fixed targets), unless `full`: then the
    tokens move with the encoder too (the loss a token-space progress signal would see; audit_b: first order)."""
    t = TR.tensors(chunk, DEV)
    fix = t["fixation"]
    with TR.autocast(DEV, cfg):
        Pc = model.perception(t["frames"][:, P:], fix[:, P:])
        with torch.set_grad_enabled(full and torch.is_grad_enabled()):
            Pp = model.perception(t["frames"][:, :P], fix[:, :P])
        emb = model.belief.percept_embed(Pc)
        y, _ = model.belief(model.belief.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:]), carry)
        b_pred = torch.cat([cache.to(y.dtype), y[:, 1::2][:, :-1]], 1)
        x0 = tuple(torch.cat([a, b if full else b.detach()], 1).float() for a, b in zip(Pp.parts(), Pc.parts()))
        sig = torch.full(fix.shape[:2], cfg["edm"]["inference_context_sigma"], device=DEV)
        sig[:, -1] = TR.sample_sigma((len(sig),), cfg, DEV, gen)
        noise = tuple(torch.randn(a.shape, device=DEV, generator=gen) for a in x0)
        c_skip, c_out, c_in, c_noise = M.precondition(sig, cfg["edm"]["sigma_data"])
        xs = tuple(a + M._bcast(sig, a) * z for a, z in zip(x0, noise))
        Fo = model.denoiser(tuple(M._bcast(c_in, a) * a for a in xs), c_noise, t["prev_action"], model.belief.memory(b_pred),
                            fix, t["episode"], t["valid"])
        se = ((Fo[0].float() - (x0[0] - M._bcast(c_skip, x0[0]) * xs[0]) / M._bcast(c_out, x0[0])) ** 2)[:, -1]
        h = model.outcome.features(b_pred[:, -1], emb[:, -1])
        reward = -model.outcome.reward_logp(h, t["reward"][:, -1]).mean()
    r = torch.arange(len(se), device=DEV)
    return {"counter": se[r, cells[:, 0]].mean(), "faced": se[r, cells[:, 1]].mean(), "reward": reward, "bulk": se.mean()}


def drift(prev, tok, odd, cls_cell):
    """Per token group: the least-squares affine map from the previous checkpoint's tokens to this one's (shared over
    the group's tokens, fit on even episodes), its residual per element on odd episodes; the residual per class."""
    groups = {"map": (lambda d: d["p"][:, :63]), "hud": (lambda d: d["p"][:, 63:]), "glimpse": (lambda d: d["g"]),
              "lighting": (lambda d: d["v"][:, None])}
    out, resid = {}, {}
    ev = torch.as_tensor(~odd)
    for name, f in groups.items():
        X, Y = f(prev).float(), f(tok).float()
        Xa = torch.cat([X, torch.ones_like(X[..., :1])], -1).double()
        A = torch.linalg.solve(Xa[ev].flatten(0, -2).T @ Xa[ev].flatten(0, -2) + 1e-6 * torch.eye(Xa.shape[-1], dtype=torch.float64),
                               Xa[ev].flatten(0, -2).T @ Y[ev].double().flatten(0, -2))
        R = (Y[~ev].double() - Xa[~ev] @ A).square().mean(-1)            # [odd frames, tokens]
        resid[name] = R.float()
        out[name] = {"residual_se": float(R.mean()), "raw_change_se": float((Y[~ev] - X[~ev]).square().mean()),
                     "token_variance": float(Y[~ev].var(0).mean())}
    R = torch.cat([resid["map"], resid["hud"]], 1)                      # [odd frames, 81]
    co = torch.as_tensor(cls_cell[odd].astype(np.int64))
    by_class = {name: float(R[co == k].mean()) for k, name in enumerate(LA.CLASSES) if (co == k).any()}
    by_class["lighting"] = float(resid["lighting"].mean())
    return out, by_class


def b1(model, cfg, Ld, cls_cell, prev):
    """Class losses, each class's copy-previous token error and the drift residual from the previous checkpoint."""
    enc = S.encode(model, Ld, cfg)
    res = {"class_losses": EV.class_losses(model, enc, Ld, cfg, n_win=10 ** 6)}
    tok = {k: torch.cat([r[k] for r in enc]).float() for k in ("p", "g", "v")}
    t_of = np.concatenate([np.arange(T + 1) for T in Ld["ep_T"]])
    odd = np.concatenate([np.full(T + 1, e % 2 == 1) for e, T in enumerate(Ld["ep_T"])])
    later = torch.as_tensor(t_of[1:] > 0)
    step = (tok["p"][1:] - tok["p"][:-1]).square().mean(-1)[later]      # [frames t >= 1, 81]
    cc = torch.as_tensor(cls_cell[1:][t_of[1:] > 0].astype(np.int64))
    res["copy_se"] = {name: float(step[cc == k].mean()) for k, name in enumerate(LA.CLASSES) if (cc == k).any()}
    res["copy_se"]["lighting"] = float((tok["v"][1:] - tok["v"][:-1]).square().mean(-1)[later].mean())
    if prev is not None:
        res["drift"], res["drift_by_class"] = drift(prev, tok, odd, cls_cell)
    return res, {k: v.half() for k, v in tok.items()}


def geometry(model, ck, cfg):
    """Parameters (harmoniser excluded), per-module slices of the flat vector, the checkpoint's Adam step per parameter
    (`pre`, linear at v_hat) and `adam_step` (LESS's Gamma for one training step on a batch with gradient g)."""
    names = [n for n, _ in model.named_parameters() if not n.startswith("harmoniser.")]
    params = [p for n, p in model.named_parameters() if not n.startswith("harmoniser.")]
    mod = np.array([MODULES.index(n.split(".")[0]) for n in names])            # modules are contiguous in this order
    assert (np.diff(mod) >= 0).all()
    bounds = np.r_[0, np.cumsum([p.numel() for p in params])]
    slices = [(int(bounds[np.flatnonzero(mod == m)[0]]), int(bounds[np.flatnonzero(mod == m)[-1] + 1])) for m in range(len(MODULES))]
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    opt.load_state_dict(ck["opt"])
    lr_of = {id(p): g["lr"] for g in opt.param_groups for p in g["params"]}
    b1_, b2_ = o["betas"]
    lrv = torch.cat([torch.full((p.numel(),), lr_of[id(p)]) for p in params]).to(DEV)
    v = torch.cat([opt.state[p]["exp_avg_sq"].float().reshape(-1) for p in params]).to(DEV)        # raw second moments
    m = torch.cat([opt.state[p]["exp_avg"].float().reshape(-1) for p in params]).to(DEV)           # raw first moments
    step = float(opt.state[params[0]]["step"])
    del opt
    pre = lrv / ((v / (1 - b2_ ** step)).sqrt() + o["eps"])          # Adam, linear at the checkpoint's v_hat
    bc1, bc2 = 1 - b1_ ** (step + 1), 1 - b2_ ** (step + 1)
    rest = lrv * b1_ * m / bc1 / ((b2_ * v / bc2).sqrt() + o["eps"])  # the next update with a zero gradient (momentum alone)

    def adam_step(g):
        """LESS's Gamma for one training step on this chunk: Adam's update with the gradient clipped as training clips it
        entering both moments, minus the momentum-alone update `rest` (common to every condition)."""
        G = g * min(1.0, o["grad_clip"] / (float(g.norm()) + 1e-6))     # torch's clip_grad_norm_ coefficient
        return lrv * (b1_ * m + (1 - b1_) * G) / bc1 / (((b2_ * v + (1 - b2_) * G ** 2) / bc2).sqrt() + o["eps"]) - rest
    return params, slices, pre, adam_step


def b2(model, cfg, ck, train, dev, targets, n_uniform, seed):
    """Per class: dots <w g, g_T> per module and geometry (euclid, adam_linear, adam_step) of each condition's gradient
    (counter / faced / control / uniform draws) with every target gradient; the targets' Gram; norms of the
    condition-minus-uniform difference in each geometry."""
    params, slices, pre, adam_step = geometry(model, ck, cfg)
    model.train()
    dots = lambda u: torch.stack([Tm[:, a:b] @ u[a:b] for a, b in slices], -1)              # [targets, modules]

    pd = prepass(model, cfg, dev["data"], dev["cuts"])
    tnames = [f"{c}_{k}_{h}" for c in [*CLASSES, "bulk"] for h in range(2) for k in (KINDS if c != "bulk" else ("bulk",))]
    Tm = torch.zeros(len(tnames), len(pre), device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for c in [*CLASSES, "bulk"]:
        for h, rows in enumerate(dev["rows"][c]):
            kinds = KINDS if c != "bulk" else ("bulk",)
            for i in range(0, len(rows), ROWS):
                part = rows[i:i + ROWS]
                chunk, carry, cache = batch(dev["data"], pd, part, -2 * P + 1, 1, -P + 1)
                cells = torch.as_tensor([dev["cells"][c][e, t] for e, t in part], device=DEV)
                state = gen.get_state()                                  # the same draws for the fixed and full targets
                terms = target_terms(model, cfg, chunk, carry, cache, cells, gen)
                fixed = [k for k in kinds if k != "counter_full"]
                for j, k in enumerate(fixed):                           # one graph at a time (6 GB card)
                    g = torch.autograd.grad(terms[k], params, retain_graph=j < len(fixed) - 1, allow_unused=True)
                    Tm[tnames.index(f"{c}_{k}_{h}")] += flat(g, params) * len(part) / len(rows)
                del terms
                if "counter_full" in kinds:
                    gen.set_state(state)
                    full = target_terms(model, cfg, chunk, carry, cache, cells, gen, full=True)["counter"]
                    g = torch.autograd.grad(full, params, allow_unused=True)
                    Tm[tnames.index(f"{c}_counter_full_{h}")] += flat(g, params) * len(part) / len(rows)
                    del full
    del pd
    gram = torch.stack([torch.stack([torch.stack([Tm[:, a:b] @ (Tm[i, a:b] * w[a:b]) for i in range(len(Tm))], 1) for a, b in slices], -1)
                        for w in (torch.ones_like(pre), pre)], -1).cpu()          # [targets, targets, modules, (euclid, adam_linear)]
    pt = prepass(model, cfg, train["data"], train["cuts"])
    out = {"targets": tnames, "gram": gram.numpy(), "modules": MODULES, "target_norms": Tm.norm(dim=1).cpu().numpy()}
    for ci, c in enumerate(CLASSES):
        D_ = np.zeros((len(train["rows"][c]), len(CONDS) + n_uniform, len(tnames), len(MODULES), len(GEOMETRIES)), np.float32)
        norms = np.zeros((len(train["rows"][c]), len(CONDS), len(GEOMETRIES)), np.float32)
        for gi, rows in enumerate(train["rows"][c]):
            rng = np.random.default_rng([seed, ci, gi])
            fixes = [np.array([train["windows"][c][k][e, t] for e, t in rows]) for k in CONDS]
            fixes += [D.fixation_prior(rng, (len(rows),)) for _ in range(n_uniform)]
            parts = [(i, batch(train["data"], pt, rows[i:i + ROWS], -P, H, 0)) for i in range(0, len(rows), ROWS)]
            G = []
            for fx in fixes:
                g = torch.zeros(len(pre), device=DEV)
                for i, (chunk, carry, cache) in parts:                 # the group's mean gradient, ROWS rows at a time
                    ch = dict(chunk, fixation=chunk["fixation"].copy())
                    ch["fixation"][:, P] = fx[i:i + ROWS]
                    torch.manual_seed(seed * 1000 + ci * 100 + gi * 10 + i)   # the same draws in every condition
                    terms, _, _ = TR.p1_losses(model, ch, carry, cache, targets, cfg)
                    loss = model.harmoniser(TR.groups(terms)) + terms.get("noise", 0)
                    g += flat(torch.autograd.grad(loss, params, allow_unused=True), params) * len(ch["first"]) / len(rows)
                G.append(g.cpu())
            u = torch.stack(G[len(CONDS):]).mean(0).to(DEV)
            su = torch.stack([adam_step(g.to(DEV)) for g in G[len(CONDS):]]).mean(0)
            for j, gc in enumerate(G):
                g = gc.to(DEV)
                sg = adam_step(g)
                D_[gi, j] = torch.stack([dots(g), dots(pre * g), dots(sg)], -1).cpu().numpy()
                if j < len(CONDS):
                    norms[gi, j] = [float((g - u).norm()), float((pre * (g - u)).norm()), float((sg - su).norm())]
            k0 = tnames.index(f"{c}_counter_0")
            print(c, "group", gi, "counter - uniform, own counter target (euclid, adam_linear, adam_step)",
                  (D_[gi, 0, k0].sum(0) - D_[gi, len(CONDS):, k0].sum(1).mean(0)).round(5).tolist(), flush=True)
        out[c] = {"dots": D_, "delta_norms": norms}
    model.eval()
    return out


def setup(cfg, split, guided, smoke, rng):
    """Rows, cut points and episode arrays for B2: train contexts (the world's training gaze) and fresh dev frames."""
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    Lt = dict(np.load(S.LABELS / "s0_train400_labels.npz"))
    where = {(e.shard, e.slot): i for i, e in enumerate(split["train"])}
    keep = np.array([(int(s), int(o)) in where for s, o in zip(Lt["ep_shard"], Lt["ep_slot"])])
    part = {e: r % 6 for r, e in enumerate(np.flatnonzero(keep))}
    fx_t = gaze(Lt, cfg, PM.window_of_state(Lt, rule) if guided else None)
    n_ctx, n_fresh = (2, 2) if smoke else (16, 64)
    groups = 2 if smoke else 6
    s_off = np.r_[0, np.cumsum(Lt["ep_T"] + 1)]
    d = FC.DIRS[Lt["s_dir"]]
    train = {"rows": {}, "windows": {}, "cuts": {}, "data": {}}
    for c, (_, cell, _) in CLASSES.items():
        rows = pick(classes_at(Lt, c, P, H - 1, keep), part, n_ctx, 6, rng)[:groups]
        train["rows"][c] = rows
        col = min(max((cell - 63) % 9, 1), 7)
        train["windows"][c] = {"counter": {}, "faced": {}, "control": {}}
        for e, t in sum(rows, []):
            g = s_off[e] + t - 1
            train["windows"][c]["counter"][e, t] = (7, col)
            train["windows"][c]["faced"][e, t] = (3 + d[g, 0], 4 + d[g, 1])
            train["windows"][c]["control"][e, t] = (1, 1)
            train["cuts"].setdefault(e, set()).add(t)
    offsets = D.frame_offsets(split["train"])
    for e, cuts in train["cuts"].items():
        last = max(cuts) + H - 1
        train["data"][e] = load(split["train"], where[(int(Lt["ep_shard"][e]), int(Lt["ep_slot"][e]))], fx_t[e], last, offsets)
    Ld = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 30 if smoke else 400)
    fx_d = gaze(Ld, cfg)
    s_off = np.r_[0, np.cumsum(Ld["ep_T"] + 1)]
    d = FC.DIRS[Ld["s_dir"]]
    half = {e: e % 2 for e in range(len(Ld["ep_T"]))}
    dev = {"rows": {}, "cells": {}, "cuts": {}, "data": {}}
    allk = np.ones(len(Ld["ep_T"]), bool)
    for c, (_, cell, _) in CLASSES.items():
        dev["rows"][c] = pick(classes_at(Ld, c, 2 * P - 1, 0, allk), half, n_fresh, 2, rng)
    bulk = [(e, int(t)) for e in range(len(Ld["ep_T"])) for t in rng.choice(np.arange(2 * P - 1, Ld["ep_T"][e] + 1), 2)
            if Ld["ep_T"][e] >= 2 * P]
    dev["rows"]["bulk"] = pick(bulk, half, n_fresh, 2, rng, cap=2)
    for c in [*CLASSES, "bulk"]:
        cell = CLASSES[c][1] if c in CLASSES else 0
        dev["cells"][c] = {}
        for e, t in sum(dev["rows"][c], []):
            g = s_off[e] + t - 1
            dev["cells"][c][e, t] = (cell, (3 + d[g, 0]) * 9 + 4 + d[g, 1])
            dev["cuts"].setdefault(e, set()).add(t - P + 1)
    eps = S.episodes(Ld)
    for e, cuts in dev["cuts"].items():
        dev["data"][e] = load(eps, e, fx_d[e], max(cuts) + P - 1)
    return train, dev


def summary(out_dir):
    """PLAN B readings from the saved arrays (CPU). B2 transfer of a condition to a target: per episode-disjoint group,
    its dot minus the uniform draws' mean dot, summed over modules, averaged over the target's two halves; mean and
    SE over groups, and the half-to-half spread (target sampling) beside it. Scales: the ordinary look's transfer
    (uniform mean) and direct training on the target (cross-half Gram under P). B1: per class and interval, the
    relative loss change, the relative change of the class's own token step and the drift residual over that step."""
    res = {}
    for f in sorted(out_dir.glob("b2_*.npz")):
        w, k = f.stem.split("_")[1], int(f.stem.split("_")[2])
        z = np.load(f)
        tn = list(z["targets"])
        gram = z["gram"].sum(2)                                         # [targets, targets, (euclid, adam_linear)]
        row = {}
        for c in CLASSES:
            dots = z[f"{c}_dots"]                                       # [groups, conds + uniform, targets, modules, geometries]
            nc = len(CONDS)
            delta = dots[:, :nc] - dots[:, nc:].mean(1, keepdims=True)  # [groups, conds, targets, modules, geometries]
            ordinary = dots[:, nc:].mean(1)                             # [groups, targets, modules, geometries]
            row[c] = {}
            for tc in [*CLASSES, "bulk"]:
                for kind in (KINDS if tc != "bulk" else ("bulk",)):
                    i0, i1 = tn.index(f"{tc}_{kind}_0"), tn.index(f"{tc}_{kind}_1")
                    r = {"target_half_reliability": float(gram[i0, i1, 0] / np.sqrt(max(gram[i0, i0, 0] * gram[i1, i1, 0], 1e-300)))}
                    for q, geo in enumerate(GEOMETRIES):
                        o_ = ordinary[:, [i0, i1], :, q].sum(-1).mean(1)
                        e = {"direct": float(gram[i0, i1, q]) if q < 2 else None,
                             "ordinary": {"est": float(o_.mean()), "se": float(o_.std(ddof=1) / np.sqrt(len(o_)))}}
                        for j, cond in enumerate(CONDS):
                            s = delta[:, j, [i0, i1], :, q].sum(-1)         # [groups, halves]
                            e[cond] = {"est": float(s.mean()), "se_groups": float(s.mean(1).std(ddof=1) / np.sqrt(len(s))),
                                       "halves": s.mean(0).tolist(),
                                       "by_module": dict(zip(MODULES, delta[:, j, [i0, i1], :, q].mean((0, 1)).tolist()))}
                        r[geo] = e
                    row[c][f"{tc}_{kind}"] = r
        res.setdefault(w, {})[k] = row
    b1j = json.loads((out_dir / "b1.json").read_text())
    drift = {}
    for w, cks in b1j.items():
        ks = sorted(cks, key=int)
        for a, b in zip(ks[:-1], ks[1:]):
            A_, B_ = cks[a]["b1"], cks[b]["b1"]
            d = {}
            for name, v in B_["class_losses"].items():
                l0, l1 = A_["class_losses"][name]["se_per_element"], v["se_per_element"]
                e = {"dlog_loss": float(np.log(l1 / l0))}
                if name in B_["copy_se"]:
                    s0, s1 = A_["copy_se"][name], B_["copy_se"][name]
                    e["dlog_step"] = float(np.log(s1 / s0))                     # dlog loss = dlog(loss / step) + dlog step
                    e["dlog_loss_over_step"] = e["dlog_loss"] - e["dlog_step"]
                    if name in B_.get("drift_by_class", {}):
                        e["drift_residual_over_step"] = B_["drift_by_class"][name] / s1
                    if name not in CHANGING:
                        e["contaminated"] = "not tested (static: no step to predict)"
                    elif abs(e["dlog_loss"]) < 0.02:
                        e["contaminated"] = "not tested (no measurable loss change)"
                    else:
                        e["contaminated"] = bool(abs(e["dlog_step"]) >= 0.5 * abs(e["dlog_loss"])
                                                 or e.get("drift_residual_over_step", 0) >= 1)
                d[name] = e
            d["groups"] = B_.get("drift", {})
            drift.setdefault(w, {})[f"{a}->{b}"] = d
    (out_dir / "summary.json").write_text(json.dumps({"b2": res, "b1": drift}, indent=1))
    return res, drift


def main():
    if sys.argv[1] == "--summary":
        summary(Path(sys.argv[2]))
        return
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    if DEV.type == "cpu":
        S.DEV = EV.DEV = DEV
    cfg0 = TR.load_cfg()
    split = D.load_split(cfg0)
    targets = np.load(TR.ROOT / cfg0["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    Ld1 = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 6 if smoke else 150)
    cls_cell = LA.cell_classes(Ld1)
    res = {}
    for w, (run, guided) in WORLDS.items():
        ck_ids = (9, 10) if smoke else range(1, 11)
        train, dev = None, None
        prev, res[w] = None, {}
        for k in ck_ids:
            path = run / f"ckpt_{k:02d}.pt"
            ck = torch.load(path, weights_only=False, map_location="cpu")
            cfg = ck["cfg"]
            model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
            if DEV.type == "cpu":
                for m in model.belief.mixers:
                    m.backend = "reference"
            if train is None:
                train, dev = setup(cfg, split, guided, smoke, np.random.default_rng(2031))
            r1, prev = b1(model, cfg, Ld1, cls_cell, prev)
            r2 = b2(model, cfg, ck, train, dev, targets, 1 if smoke else 4, 2031)
            np.savez_compressed(out_dir / f"b2_{w}_{k:02d}.npz", targets=np.array(r2["targets"]), gram=r2["gram"],
                                target_norms=r2["target_norms"],
                                **{f"{c}_{x}": r2[c][x] for c in CLASSES for x in ("dots", "delta_norms")})
            res[w][k] = {"checkpoint": str(path), "sha256": TR.file_sha256(path), "update": ck["state"]["update"],
                         "harmoniser_sigma": ck["model"]["harmoniser.log_sigma"].exp().tolist(), "b1": r1}
            (out_dir / "b1.json").write_text(json.dumps(res, indent=1))
            print(w, k, "B1 class losses", {n: round(v["se_per_element"], 4) for n, v in r1["class_losses"].items()},
                  "peak GiB", round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if DEV.type == "cuda" else None, flush=True)
            del model, ck
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
        json.dump({"rows": {c: train["rows"][c] for c in CLASSES}, "fresh": {c: dev["rows"][c] for c in [*CLASSES, "bulk"]}},
                  open(out_dir / f"rows_{w}.json", "w"))
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
