"""Readers of the v3 evaluation, fitted per checkpoint on its own encoder's REAL train-400 tokens and qualified on its
REAL dev tokens (the encoder trains in v3, so no reader carries over between checkpoints).
- Periphery HUD values (health, drink, sapling, wood): nearest value mean (v2's reader).
- Glimpse HUD values: one small MLP per (counter cell, window centre) on the cell's own glimpse token, qualified per
  centre; rows at an unqualified centre are not read. (The fovea's last 3 x 3 conv mixes neighbouring cells, so a
  token varies with its neighbours and its slot. Tried first on the base checkpoint and rejected, before any v3
  model was read: one MLP per cell pooled over centres with a slot one-hot (wood 0.963, health 0.979), and one on
  the whole window's 9 tokens (drink 0.905, wood 0.885).)
- Table: an MLP "table or not" on the periphery token + lighting, and one on the glimpse token + slot.
Qualification: accuracy >= 0.99 on dev real tokens, overall and in every window centre with >= 100 dev rows (tables:
recall and non-table specificity >= 0.99 on the faced cells). Support: a token is READ only if its distance to the
real mean of the predicted value (at that slot, for the glimpse) is within the 99th percentile of the real tokens'
own distances; otherwise it is unreadable (reported, never counted as the consequence).
Usage (qualification report, project venv, GPU): python -m artifacts.experiments.20261030_m05_guided_fovea.instruments
  CKPT OUT_JSON
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from saccade import model as M

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
CAT, FC, PM = EV.CAT, EV.FC, EV.PM
DEV, ACC_MIN, MIN_ROWS, Q_SUPPORT = EV.DEV, 0.99, 100, 0.99
HUD = EV.HUD_CELLS


def states(enc, L):
    return np.concatenate([np.arange(len(r["p"])) + o for r, o in zip(enc, np.r_[0, np.cumsum(L["ep_T"] + 1)][:-1])])


def fit_mlp(X, y, seed=0, epochs=30, min_steps=4000):
    """[n, d] float, [n] int -> MLP d -> 128 -> classes (AdamW, batches of 2,048), deterministic for a seed; at least
    `epochs` passes and `min_steps` steps (30 epochs of a single window centre's ~8 k rows are only ~120 steps, which
    undertrained the per-centre readers: drink 0.72-0.96 on the base checkpoint)."""
    torch.manual_seed(seed)
    classes = np.unique(y)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, len(classes))).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    Xd, yd = X.float().to(DEV), torch.as_tensor(np.searchsorted(classes, y), device=DEV)
    g = torch.Generator(device=DEV).manual_seed(seed)
    steps, epoch = 0, 0
    with torch.enable_grad():
        while epoch < epochs or steps < min_steps:
            for idx in torch.randperm(len(Xd), device=DEV, generator=g).split(2048):
                opt.zero_grad()
                nn.functional.cross_entropy(net(Xd[idx]), yd[idx]).backward()
                opt.step()
                steps += 1
            epoch += 1
    net.eval()
    return {"net": net, "classes": classes}


@torch.no_grad()
def predict(rd, X):
    return rd["classes"][torch.cat([rd["net"](x.float().to(DEV)).argmax(1).cpu() for x in X.split(65536)]).numpy()]


def support(tok, val, key=None):
    """Per value (and key, e.g. slot): real mean and the Q_SUPPORT quantile of the real distances to it."""
    out = {}
    keys = np.zeros(len(val), int) if key is None else key
    for k in np.unique(keys):
        for v in np.unique(val[keys == k]):
            m = (keys == k) & (val == v)
            if m.sum() >= 30:
                mu = tok[torch.as_tensor(m)].float().mean(0)
                d = (tok[torch.as_tensor(m)].float() - mu).norm(dim=-1)
                out[(int(k), int(v))] = (mu, float(torch.quantile(d, Q_SUPPORT)))
    return out


def readable(sup, tok, pred, key=None):
    keys = np.zeros(len(pred), int) if key is None else key
    ok = np.zeros(len(pred), bool)
    for i, (k, v) in enumerate(zip(keys, pred)):
        s = sup.get((int(k), int(v)))
        ok[i] = s is not None and float((tok[i].float() - s[0]).norm()) <= s[1]
    return ok


def glimpse_rows(enc, L, cell):
    """Frames whose window covers `cell`: the window's tokens [n, 9, 64], the cell's slot [n], centre [n], state [n]."""
    s = states(enc, L)
    fix, g = torch.cat([r["fix"] for r in enc]), torch.cat([r["g"] for r in enc])
    hit = M.window(fix) == cell
    rows, slot = hit.nonzero(as_tuple=True)
    return g[rows], slot.numpy(), (fix[rows, 0] * 9 + fix[rows, 1]).numpy(), s[rows.numpy()]


def feats(tok, slot):
    return torch.cat([tok.float(), nn.functional.one_hot(torch.as_tensor(slot), 9).float()], 1)



def fit(enc_tr, L_tr, enc_dv, L_dv):
    """All readers of one checkpoint and their qualification (enc_*: guided-fixation encodings, so counters and faced
    cells are often in the glimpse; periphery tokens do not depend on the fixation)."""
    rd, qual = {"p": {}, "g": {}}, {"p": {}, "g": {}}
    s_tr, s_dv = states(enc_tr, L_tr), states(enc_dv, L_dv)
    p_tr, p_dv = torch.cat([r["p"] for r in enc_tr]), torch.cat([r["p"] for r in enc_dv])
    for name, (cell, arr, col) in HUD.items():
        v_tr, v_dv = EV.hud_value(L_tr, name, s_tr), EV.hud_value(L_dv, name, s_dv)
        means = EV.nearest_means(p_tr[:, cell], v_tr)
        known = np.isin(v_dv, list(means))
        acc = float((EV.read(means, p_dv[:, cell][known]) == v_dv[known]).mean())
        rd["p"][name] = {"means": means, "support": support(p_tr[:, cell], v_tr), "qualified": acc >= ACC_MIN}
        qual["p"][name] = {"accuracy": acc, "rows": int(known.sum()), "qualified": acc >= ACC_MIN}
        (gt, sl, ce, st), (gd, sd, cd, sdv) = glimpse_rows(enc_tr, L_tr, cell), glimpse_rows(enc_dv, L_dv, cell)
        vt, vd = EV.hud_value(L_tr, name, st), EV.hud_value(L_dv, name, sdv)
        gt, gd = gt[np.arange(len(sl)), sl], gd[np.arange(len(sd)), sd]          # the cell's own token
        for c in np.unique(ce):
            mt, md = ce == c, cd == c
            if mt.sum() < 200:
                continue
            r = fit_mlp(gt[torch.as_tensor(mt)], vt[mt])
            ok = md & np.isin(vd, r["classes"])
            acc = float((predict(r, gd[torch.as_tensor(ok)]) == vd[ok]).mean()) if ok.sum() >= MIN_ROWS else None
            q = acc is not None and acc >= ACC_MIN
            rd["g"][(name, int(c))] = {**r, "support": support(gt[torch.as_tensor(mt)], vt[mt]), "qualified": q}
            qual["g"][f"{name}@{int(c)}"] = {"accuracy": acc, "rows": int(ok.sum()), "qualified": q}
    # tables
    def table_label(L, s, cells):                                       # HUD cells (>= 63) are never tables
        c = np.minimum(cells, 62)
        return (L["s_view"][s, c // 9, c % 9] == CAT.TABLE) & (cells < 63)
    rng = np.random.default_rng(2050)
    v_tr = torch.cat([r["v"] for r in enc_tr])
    map_cells = np.arange(63)
    lab = table_label(L_tr, np.repeat(s_tr, 63), np.tile(map_cells, len(s_tr))).reshape(len(s_tr), 63)
    pos = np.argwhere(lab); neg = np.argwhere(~lab)
    neg = neg[rng.choice(len(neg), min(len(neg), 20 * len(pos)), replace=False)]
    idx = np.concatenate([pos, neg])
    Xp = torch.cat([p_tr[idx[:, 0], idx[:, 1]].float(), v_tr[idx[:, 0]].float()], 1)
    rt = fit_mlp(Xp, lab[idx[:, 0], idx[:, 1]].astype(int), epochs=8, min_steps=0)
    d = FC.DIRS[L_dv["s_dir"][s_dv]]
    fcell = (3 + d[:, 0]) * 9 + 4 + d[:, 1]
    is_t = table_label(L_dv, s_dv, fcell)
    v_dv = torch.cat([r["v"] for r in enc_dv])
    pred = predict(rt, torch.cat([p_dv[np.arange(len(s_dv)), fcell].float(), v_dv.float()], 1)) == 1
    rec, spec = float(pred[is_t].mean()), float((~pred[~is_t]).mean())
    sup_p = support(p_tr[idx[:, 0], idx[:, 1]], lab[idx[:, 0], idx[:, 1]].astype(int))
    rd["table_p"] = {**rt, "support": sup_p, "qualified": rec >= ACC_MIN and spec >= ACC_MIN}
    qual["table_p"] = {"recall": rec, "specificity": spec, "tables": int(is_t.sum()), "qualified": rd["table_p"]["qualified"]}
    fix_tr, g_tr = torch.cat([r["fix"] for r in enc_tr]), torch.cat([r["g"] for r in enc_tr])
    win_tr = M.window(fix_tr).numpy()
    labg = table_label(L_tr, np.repeat(s_tr, 9), win_tr.reshape(-1)).reshape(len(s_tr), 9)
    pos = np.argwhere(labg); neg = np.argwhere(~labg)
    neg = neg[rng.choice(len(neg), min(len(neg), 20 * len(pos)), replace=False)]
    idx = np.concatenate([pos, neg])
    rg = fit_mlp(feats(g_tr[idx[:, 0], idx[:, 1]], idx[:, 1]), labg[idx[:, 0], idx[:, 1]].astype(int), epochs=8, min_steps=0)
    fix_dv, g_dv = torch.cat([r["fix"] for r in enc_dv]), torch.cat([r["g"] for r in enc_dv])
    win_dv = M.window(fix_dv).numpy()
    inw = win_dv == fcell[:, None]
    rows, slot = np.nonzero(inw)
    is_tg = is_t[rows]
    predg = predict(rg, feats(g_dv[rows, slot], slot)) == 1
    recg, specg = float(predg[is_tg].mean()), float((~predg[~is_tg]).mean())
    rd["table_g"] = {**rg, "support": support(g_tr[idx[:, 0], idx[:, 1]], labg[idx[:, 0], idx[:, 1]].astype(int), idx[:, 1]),
                     "qualified": recg >= ACC_MIN and specg >= ACC_MIN}
    qual["table_g"] = {"recall": recg, "specificity": specg, "tables": int(is_tg.sum()), "faced_in_window": int(len(rows)),
                       "qualified": rd["table_g"]["qualified"]}
    return rd, qual


def main():
    from saccade.eval import s1 as S
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck.get("cfg") or torch.load(EV.BASE, weights_only=False, map_location="cpu")["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    load = lambda f: dict(np.load(EV.ROOT / f"artifacts/eda/s0/{f}"))
    L_tr, L_dv = load("s0_train400_labels.npz"), load("s0_dev_labels.npz")
    enc_tr = EV.encode_with(model, L_tr, cfg, PM.window_of_state(L_tr, rule))
    enc_dv = EV.encode_with(model, L_dv, cfg, PM.window_of_state(L_dv, rule))
    _, qual = fit(enc_tr, L_tr, enc_dv, L_dv)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qual, indent=1, default=str))
    print(json.dumps(qual, indent=1, default=str))


if __name__ == "__main__":
    main()
