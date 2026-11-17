"""F6 dissection, third pass (PLAN.md): what the denoiser's loss is spent on and where it uses the memory.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.g1map map [checkpoints...]
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.g1map pull [checkpoints...]
"""
import importlib
import json
import os
import sys

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

DZ = importlib.import_module("artifacts.experiments.20261114_f6_dissection.dose")
OUT, DEV = DZ.OUT, DZ.DEV
GROUPS = ("map", "hud_digits", "hud_other", "glimpse", "lighting")
EVENTS = ("health_down", "health_up", "food_down", "drink_down", "energy_down_awake", "energy_up_asleep")
DIGIT = {"health": 63, "food": 64, "drink": 65, "energy": 66}
SEED, N_UNIFORM, N_EVENT, GAP = 20261114, 3000, 300, 32


def targets(L):
    """(episode, t, tag): uniform frames and event frames (t = the state after the transition), t >= 9 with a partner
    window at least GAP frames away."""
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rng = np.random.default_rng(SEED)
    ok, ev = [], {n: [] for n in EVENTS}
    for e, T in enumerate(L["ep_T"]):
        if T < 2 * GAP + 20:
            continue
        body, sleep = L["s_body"][off[e]:off[e + 1]].astype(int), L["s_sleep"][off[e]:off[e + 1]].astype(bool)
        d = body[1:] - body[:-1]
        ts = np.arange(9, T + 1)
        ok += [(e, int(t)) for t in ts]
        flags = {"health_down": d[:, 0] < 0, "health_up": d[:, 0] > 0, "food_down": d[:, 1] < 0, "drink_down": d[:, 2] < 0,
                 "energy_down_awake": (d[:, 3] < 0) & ~sleep[:-1], "energy_up_asleep": (d[:, 3] > 0) & sleep[:-1]}
        for n, f in flags.items():
            ev[n] += [(e, int(t)) for t in ts if f[t - 1]]
    out = [(*ok[i], "uniform") for i in rng.choice(len(ok), N_UNIFORM, replace=False)]
    for n in EVENTS:
        out += [(*ev[n][i], n) for i in rng.choice(len(ev[n]), min(N_EVENT, len(ev[n])), replace=False)]
    return sorted(out), off


def episode(model, cfg, ep):
    rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
    fx = [D.fixation_prior(rng, ()) for _ in range(ep.T + 1)]
    c = TR.tensors(D.gather([ep], [[(0, 1, t, *fx[t]) for t in range(ep.T + 1)]]), DEV)
    with torch.no_grad(), TR.autocast(DEV, cfg):
        Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                        for a in range(0, ep.T + 1, 512)])), c["fixation"])
        y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(Pe), c["reward"], c["terminated"], c["action"]),
                            model.belief.initial(1, DEV))
        b_pred = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
    return c, Pe, b_pred


def forward(model, cfg, c, Pe, beliefs, t, seed):
    """The flow term on target frame t at every signal level: per-row errors by part, the ramp weights, the library's
    own per-frame loss. beliefs: [rows, P + 1, W] (rows = levels x variants, level-major)."""
    sc, P = cfg["edm"]["shortcut"], cfg["data"]["context_prefix"]
    k = sc["k_max"]
    rows = beliefs.shape[0]
    sl = slice(t - P, t + 1)
    rep = lambda a: a[:, sl].expand(rows, *a[:, sl].shape[1:])
    x0 = tuple(rep(a) for a in Pe.parts())
    g = torch.Generator(device=DEV).manual_seed(seed)
    one = tuple(torch.randn((1, *a.shape[1:]), device=DEV, generator=g) for a in x0)
    noise = tuple(n.expand(rows, *n.shape[1:]) for n in one)                # the same draw for every level and variant
    sg = torch.full((rows, P + 1), sc["context_signal"], device=DEV, dtype=torch.long)
    sg[:, -1] = torch.arange(k, device=DEV).repeat_interleave(rows // k)
    lv = [(sg, torch.full_like(sg, k.bit_length() - 1))] * 2
    args = (rep(c["prev_action"]), model.belief.memory(beliefs), rep(c["fixation"]), rep(c["episode"]), rep(c["valid"]))
    tau = sg.float() / k
    zt = tuple((1 - M._bcast(tau, x)) * n + M._bcast(tau, x) * x for x, n in zip(x0, noise))
    code = tuple(M.shortcut_code(sg, lv[0][1], k) for _ in range(2))
    out = model.denoiser(zt, code, *args)
    err = [(o.float() - x.float()) ** 2 for o, x in zip(out, x0)]           # p [rows, n, 81, 16], g [rows, n, 9, 64], v [rows, n, 8]
    lib = M.shortcut_loss(model.denoiser, x0, *args, sc, torch.zeros(rows, dtype=torch.bool, device=DEV), False, lv, noise)
    return [e[:, -1] for e in err], sc["ramp"][0] * tau[:, -1] + sc["ramp"][1], lib[:, -1].float()


@torch.no_grad()
def run_map(k):
    path = OUT / f"g1map_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, _ = DZ.load(k)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    tg, off = targets(L)
    eps = S.episodes(L)
    P, K = cfg["data"]["context_prefix"], cfg["edm"]["shortcut"]["k_max"]
    rng = np.random.default_rng(SEED + 1)
    rec, cur, check = [], None, 0.
    for i, (e, t, tag) in enumerate(tg):
        if cur != e:
            c, Pe, b_pred = episode(model, cfg, eps[e])
            cur = e
        T = int(L["ep_T"][e])
        far = [o for o in range(P, T + 1) if abs(o - t) >= GAP]
        o = int(rng.choice(far))
        bel = torch.stack([b_pred[0, t - P:t + 1], b_pred[0, o - P:o + 1]]).repeat(K, 1, 1)     # level-major: (true, other) per level
        with TR.autocast(DEV, cfg):
            (ep_, eg, ev), w, lib = forward(model, cfg, c, Pe, bel, t, SEED * 1_000_003 + e * 4_099 + t)
        cell = ep_.mean(-1)                                                    # [rows, 81]
        parts = torch.stack([cell[:, :63].mean(1), cell[:, 63:67].mean(1), cell[:, 67:].mean(1), eg.mean((1, 2)), ev.mean(1)], 1)
        mine = (w * (ep_.flatten(1).sum(1) + eg.flatten(1).sum(1) + ev.flatten(1).sum(1)) / (81 * 16 + 9 * 64 + 8))
        check = max(check, float((mine - lib).abs().max()))
        a = off[e]
        prev, now = Pe.p[0, t - 1].float(), Pe.p[0, t].float()
        cp = (prev - now).square().mean(-1)
        rec.append({"tag": tag, "moved": bool((L["s_pos"][a + t] != L["s_pos"][a + t - 1]).any()), "asleep": bool(L["s_sleep"][a + t - 1]),
                    "night": bool(L["s_light"][a + t] < .5), "parts": parts.reshape(K, 2, 5).cpu().numpy(),
                    "digits": cell[:, 63:67].reshape(K, 2, 4).cpu().numpy(),
                    "copy": np.array([float(cp[:63].mean()), float(cp[63:67].mean()), float(cp[67:].mean())]),
                    "copy_digits": cp[63:67].cpu().numpy()})
        if (i + 1) % 500 == 0:
            print(k, i + 1, len(tg), "known-answer max abs", check, flush=True)
    w = cfg["edm"]["shortcut"]["ramp"][0] * np.arange(K) / K + cfg["edm"]["shortcut"]["ramp"][1]
    size = np.array([63 * 16, 4 * 16, 14 * 16, 9 * 64, 8]) / 1880                 # each group's share of the 1,880 scalars
    res = {"checkpoint": k, "targets": len(rec), "known_answer_max_abs": check, "levels_tau": (np.arange(K) / K).tolist(),
           "ramp": w.tolist(), "group_scalar_share": dict(zip(GROUPS, size.tolist())), "classes": {}, "events": {}}
    uni = [r for r in rec if r["tag"] == "uniform"]
    classes = {"all": lambda r: True, "moved": lambda r: r["moved"] and not r["asleep"], "still_awake": lambda r: not r["moved"] and not r["asleep"],
               "asleep": lambda r: r["asleep"], "still_awake_day": lambda r: not r["moved"] and not r["asleep"] and not r["night"],
               "still_awake_night": lambda r: not r["moved"] and not r["asleep"] and r["night"]}
    total = np.stack([r["parts"][:, 0] for r in uni])                            # [N, K, 5] true belief
    trained = (total * w[None, :, None] * size[None, None]).sum()                # the term as trained (uniform levels)
    for name, f in classes.items():
        rs = [r for r in uni if f(r)]
        if not rs:
            continue
        p = np.stack([r["parts"] for r in rs])                                   # [n, K, 2, 5]
        res["classes"][name] = {
            "frames": len(rs), "copy_previous": dict(zip(GROUPS[:3], np.mean([r["copy"] for r in rs], 0).tolist())),
            "error_true_by_level": {g_: p[:, :, 0, j].mean(0).tolist() for j, g_ in enumerate(GROUPS)},
            "other_over_true_by_level": {g_: (p[:, :, 1, j].mean(0) / np.maximum(p[:, :, 0, j].mean(0), 1e-12)).tolist() for j, g_ in enumerate(GROUPS)},
            "share_of_trained_term": {g_: float((p[:, :, 0, j] * w[None] * size[j]).sum() / trained) for j, g_ in enumerate(GROUPS)}}
    for n in EVENTS:
        rs = [r for r in rec if r["tag"] == n]
        j = list(DIGIT).index(n.split("_")[0])
        d = np.stack([r["digits"][:, :, j] for r in rs])                         # [n, K, 2]
        res["events"][n] = {"frames": len(rs), "copy_previous_at_cell": float(np.mean([r["copy_digits"][j] for r in rs])),
                            "error_true_by_level": d[:, :, 0].mean(0).tolist(), "error_other_by_level": d[:, :, 1].mean(0).tolist()}
    print(k, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


def run_pull(k):
    path = OUT / f"g1pull_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, _ = DZ.load(k)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    tg, off = targets(L)
    tg = [x for x in tg if x[2] == "uniform"][::10]
    eps = S.episodes(L)
    P, K = cfg["data"]["context_prefix"], cfg["edm"]["shortcut"]["k_max"]
    rows, cur = [], None
    for e, t, _ in tg:
        if cur != e:
            c, Pe, b_pred = episode(model, cfg, eps[e])
            cur = e
        bel = b_pred[0, t - P:t + 1].float().unsqueeze(0).repeat(K, 1, 1).requires_grad_(True)
        with TR.autocast(DEV, cfg):
            (ep_, eg, ev), w, _ = forward(model, cfg, c, Pe, bel, t, SEED * 1_000_003 + e * 4_099 + t)
        per = (ep_.flatten(1).sum(1) + eg.flatten(1).sum(1) + ev.flatten(1).sum(1)) / 1880          # unweighted, per level
        g = torch.autograd.grad(per.sum(), bel)[0]                                                  # each row is its own level
        rows.append({"all": g.flatten(1).norm(dim=1).cpu().numpy(), "target_frame": g[:, -1].norm(dim=1).cpu().numpy(),
                     "loss": per.detach().cpu().numpy(), "ramp": w.cpu().numpy()})
    m = lambda key: np.mean([r[key] for r in rows], 0)
    pull, w = m("all"), rows[0]["ramp"]
    res = {"checkpoint": k, "targets": len(rows), "levels_tau": (np.arange(K) / K).tolist(), "ramp": w.tolist(),
           "pull_unweighted_by_level": pull.tolist(), "pull_on_target_frame_belief_by_level": m("target_frame").tolist(),
           "loss_unweighted_by_level": m("loss").tolist(), "pull_as_trained_by_level": (pull * w).tolist(),
           "share_of_trained_pull_by_level": (pull * w / (pull * w).sum()).tolist(),
           "share_of_unweighted_pull_by_level": (pull / pull.sum()).tolist()}
    print(k, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    mode, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
    for k in ks or (10, 2):
        {"map": run_map, "pull": run_pull}[mode](k)
