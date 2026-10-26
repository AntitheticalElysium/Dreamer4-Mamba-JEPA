"""EDA (2026-10-27): what each periphery acuity lets a reader see, to choose the periphery's input resolution.

The periphery sees each 7 x 7 cell downsampled to r x r (torch adaptive average pooling, rounded to 8 bits: a lower
resolution image), r in {7 (full), 5, 4, 3, 2, 1}; the fovea keeps full resolution. For each r, the S0 pixel-probe
definition (MLP 256-256 GELU on the cell input + frame mean RGB; kind and terrain heads; AdamW 1e-3, batch 4096,
held-out plateau, 2 seeds) is trained on train-400 and read on all dev, plus a HUD value probe (same MLP on the pooled
HUD cell + its position one-hot; the displayed quantity 0-9). This measures what the INPUT allows a supervised reader,
not what the learned periphery will keep (a learning claim, read in S1).

DECISION RULE (declared before running; derived from agreed contracts, not tuned to HUD digits):
the coarsest r such that
  (i)  day (light >= 0.5), rings 1-4: presence recall, kind accuracy on entities and terrain accuracy within 0.01 of
       full resolution, and each kind's recall within 0.02 (the periphery keeps the what and where of map content);
  (ii) rings 1-2, day AND night: each kind's recall >= 0.95, so the S3 perception blocker (0.90 on rings 1-2) still
       tests learned erasure rather than input acuity.
HUD value accuracy and night rings 3-4 are REPORTED as the consequences, never used to choose.
Usage (project venv, GPU): eda_periphery_acuity.py OUT_JSON
"""
import json, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda")
import replay as RP
from saccade.eval import gates as G

DEV = torch.device("cuda")
RING = np.maximum(np.abs(np.arange(7)[:, None] - 3), np.abs(np.arange(9)[None] - 4)).ravel()
KEEP = RING > 0
# HUD (row, col) -> source: ("body", index) or ("inv", index); 16 cells carry a quantity
HUD = {(0, 0): ("body", 0), (0, 1): ("body", 1), (0, 2): ("body", 2), (0, 3): ("body", 3), (0, 4): ("inv", 5),
       (0, 5): ("inv", 0), (0, 6): ("inv", 1), (0, 7): ("inv", 2), (0, 8): ("inv", 3), (1, 0): ("inv", 4),
       (1, 1): ("inv", 6), (1, 2): ("inv", 7), (1, 3): ("inv", 8), (1, 4): ("inv", 9), (1, 5): ("inv", 10),
       (1, 6): ("inv", 11)}
RES = (7, 5, 4, 3, 2, 1)


def episodes(L):
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    for e, ep in enumerate(G.episode_grids(L)):
        obs = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"].numpy()
        n = len(obs); a, b = s_off[e], s_off[e + 1]
        allc = obs.reshape(n, 9, 7, 9, 7, 3).transpose(0, 1, 3, 2, 4, 5)        # [n, 9, 9, 7, 7, 3]
        mp = allc[:, :7].reshape(n, 63, 7, 7, 3)[:, KEEP].reshape(-1, 7, 7, 3)
        mean = np.repeat(np.round(obs[:, :49].reshape(n, -1, 3).mean(1)).astype(np.uint8)[:, None], KEEP.sum(), 1).reshape(-1, 3)
        kind = ep["grid"].reshape(n, 63)[:, KEEP].ravel(); terr = ep["view"].reshape(n, 63)[:, KEEP].ravel().astype(np.int64)
        terr = np.where(kind == 0, terr, -1)
        night = np.repeat(L["s_light"][a:b] < 0.5, KEEP.sum())
        hud = np.stack([allc[:, 7 + r, c] for (r, c) in HUD], 1).reshape(-1, 7, 7, 3)
        src = {"body": L["s_body"][a:b], "inv": L["s_inv"][a:b]}
        val = np.stack([np.minimum(src[s][:, i], 9) for (s, i) in HUD.values()], 1).ravel().astype(np.int64)
        pos = np.tile(np.arange(len(HUD)), n)
        yield (e, mp, mean, kind.astype(np.int64), terr, np.tile(RING[KEEP], n), night, hud, val, pos)


def pool(x, r):
    """uint8 [N, 7, 7, 3] -> float [N, r*r*3] in [0, 1]: area-downsampled, rounded to 8 bits."""
    y = x.permute(0, 3, 1, 2).float()
    if r < 7:
        y = torch.round(F.adaptive_avg_pool2d(y, r))
    return y.flatten(1) / 255


def mlp(n_in, heads):
    body = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU())
    return nn.ModuleDict({"body": body, **{k: nn.Linear(256, v) for k, v in heads.items()}}).to(DEV)


def fit(make_x, ys, n, n_val, losses, seed):
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    m = None; best, best_state, best_ep = 1e9, None, 0
    for epoch in range(30):
        if m is None:
            m = mlp(make_x(torch.arange(2, device=DEV), False).shape[1], {k: c for k, c in losses.items()})
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
        m.train()
        for idx in torch.randperm(n, device=DEV, generator=g).split(4096):
            h = m["body"](make_x(idx, False)); loss = sum(F.cross_entropy(m[k](h), ys(k, idx, False), ignore_index=-1) for k in losses)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([sum(F.cross_entropy(m[k](m["body"](make_x(i, True))), ys(k, i, True), ignore_index=-1) for k in losses).item()
                                for i in torch.arange(n_val, device=DEV).split(65536)]))
        if vl < best:
            best, best_state, best_ep = vl, {k: v.clone() for k, v in m.state_dict().items()}, epoch
        elif epoch - best_ep >= 3:
            break
    m.load_state_dict(best_state)
    return m, best_ep + 1


def main():
    Ltr, Ldev = dict(np.load("artifacts/eda/s0/s0_train400_labels.npz")), dict(np.load("artifacts/eda/s0/s0_dev_labels.npz"))
    tr, va = [[] for _ in range(9)], [[] for _ in range(9)]
    for item in episodes(Ltr):
        for p, x in zip(va if item[0] % 10 == 9 else tr, item[1:]):
            p.append(x)
    T = [torch.as_tensor(np.concatenate(p)).to(DEV) for p in tr]; V = [torch.as_tensor(np.concatenate(p)).to(DEV) for p in va]
    dev = list(episodes(Ldev))
    print("train map cells", len(T[0]), "HUD cells", len(T[6]), flush=True)
    out = {"rule": __doc__.split("DECISION RULE")[1].split("Usage")[0].strip(), "results": {}}
    for r in RES:
        res = {}
        for seed in (0, 1):
            mx = lambda idx, v: torch.cat([pool((V if v else T)[0][idx], r), (V if v else T)[1][idx].float() / 255], 1)
            my = lambda k, idx, v: (V if v else T)[{"kind": 2, "terr": 3}[k]][idx]
            m, ep_map = fit(mx, my, len(T[0]), len(V[0]), {"kind": 5, "terr": 17}, seed)
            hx = lambda idx, v: torch.cat([pool((V if v else T)[6][idx], r), F.one_hot((V if v else T)[8][idx], len(HUD)).float()], 1)
            hy = lambda k, idx, v: (V if v else T)[7][idx]
            h, ep_hud = fit(hx, hy, len(T[6]), len(V[6]), {"val": 10}, seed)
            R = {k: [] for k in ("kind", "kp", "terr", "tp", "ring", "night", "ep", "val", "vp", "pos", "hep")}
            with torch.no_grad():
                for e, mp, mean, kind, terr, ring, night, hud, val, pos in dev:
                    mpg = torch.as_tensor(mp).to(DEV); x = torch.cat([pool(mpg, r), torch.as_tensor(mean).to(DEV).float() / 255], 1)
                    z = m["body"](x)
                    hx_ = torch.cat([pool(torch.as_tensor(hud).to(DEV), r), F.one_hot(torch.as_tensor(pos).to(DEV), len(HUD)).float()], 1)
                    for k, v in (("kind", kind), ("kp", m["kind"](z).argmax(1).cpu().numpy()), ("terr", terr),
                                 ("tp", m["terr"](z).argmax(1).cpu().numpy()), ("ring", ring), ("night", night),
                                 ("ep", np.full(len(kind), e)), ("val", val), ("vp", h["val"](h["body"](hx_)).argmax(1).cpu().numpy()),
                                 ("pos", pos), ("hep", np.full(len(val), e))):
                        R[k].append(v)
            R = {k: np.concatenate(v) for k, v in R.items()}
            s = {"epochs": [ep_map, ep_hud]}
            pres, pp, valid = R["kind"] > 0, R["kp"] > 0, R["terr"] >= 0
            for lname, lm in (("day", ~R["night"]), ("night", R["night"])):
                for rname, rm in (("r1_4", R["ring"] >= 1), ("r1_2", R["ring"] <= 2), ("r3_4", R["ring"] >= 3)):
                    c = lm & rm; key = f"{lname}_{rname}"
                    s[key] = {"presence_recall": G.estimate(pp, pres & c, R["ep"]), "presence_precision": G.estimate(pres, pp & c, R["ep"]),
                              "kind_acc": G.estimate(R["kp"] == R["kind"], pres & c, R["ep"]),
                              "terrain_acc": G.estimate(R["tp"] == R["terr"], valid & c, R["ep"]),
                              **{f"recall_{kn}": G.estimate(R["kp"] == k, (R["kind"] == k) & c, R["ep"]) for k, kn in enumerate(G.KINDS, 1)}}
            s["hud_value_acc"] = G.estimate(R["vp"] == R["val"], np.ones(len(R["val"]), bool), R["hep"])
            s["hud_value_acc_by_cell"] = {f"{rc[0]},{rc[1]}": float((R["vp"] == R["val"])[R["pos"] == i].mean()) for i, rc in enumerate(HUD)}
            s["hud_value_acc_nonzero"] = G.estimate(R["vp"] == R["val"], R["val"] > 0, R["hep"])
            res[seed] = s
            print(r, seed, json.dumps({k: (round(v["est"], 4) if isinstance(v, dict) and "est" in v else v)
                                        for k, v in {"day_presence": s["day_r1_4"]["presence_recall"], "day_terr": s["day_r1_4"]["terrain_acc"],
                                                     "night_r1_2_min_kind": min(s["night_r1_2"][f"recall_{kn}"]["est"] for kn in G.KINDS),
                                                     "hud": s["hud_value_acc"]}.items()}), flush=True)
        out["results"][str(r)] = res
        json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
