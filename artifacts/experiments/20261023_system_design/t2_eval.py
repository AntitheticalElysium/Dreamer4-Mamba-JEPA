"""T2 evaluation (T2_PROTOCOL.md, measurements 1-3). Usage: t2_eval.py --data DIR --run RUN_DIR [--ckpts 0,1000,...]

DEV episodes only (never trained on). Probes are ridge regressions fitted on the even-indexed DEV episodes and scored on the
odd-indexed ones (episode-disjoint). All features are per-token LayerNorm'd latents (the space the loss uses).
  mob_ap        per type (zombie, cow, skeleton): shared linear probe on a map cell's own token + per-position bias;
                tie-grouped average precision pooled over map cells of the scoring episodes.
  terrain_acc   17-class ridge on the cell token + position bias, argmax accuracy (all cells / non-grass cells).
  vitals_r2     ridge from the 18 HUD tokens (flattened) to health, food, drink, energy; R^2.
  eff_rank      exp(entropy) of the token covariance spectrum per position, mean over the 81 positions.
  imagined      512 fixed windows (20 frames) from scoring episodes: 4 real frames, then 16 open-loop steps with the
                recorded actions. Probes fitted on REAL tokens are applied to imagined tokens at depth k and scored against
                the real labels of that frame; 'copy' applies them to the last real frame instead; 'ceiling' to the real
                frame itself. Relative latent error = |zhat - z_k|^2 / |z_3 - z_k|^2 (pooled).
References computed once per data dir: raw 7x7 pixel patches (same probes).
"""
import argparse, json, os, sys
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_joint import Encoder, Predictor, C, D_LAT

MAP = np.array([r * 9 + c for r in range(7) for c in range(9)])
HUD = np.arange(63, 81)


def ap_grouped(y, s):
    """Average precision with tied scores grouped (each distinct threshold is one step)."""
    y = np.asarray(y, bool); s = np.asarray(s, np.float64)
    P = y.sum()
    if P == 0:
        return float("nan")
    order = np.argsort(-s, kind="stable"); s, y = s[order], y[order]
    tp = np.cumsum(y); fp = np.cumsum(~y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]               # last index of each tie group
    tp, fp = tp[last], fp[last]
    rec = tp / P; prec = tp / (tp + fp)
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ridge(X, Y, lam=1e-2):
    X = np.c_[X, np.ones(len(X))]
    A = X.T @ X; A[np.diag_indices_from(A)] += lam * len(X)
    return np.linalg.solve(A, X.T @ Y)


def apply(Wt, X):
    return np.c_[X, np.ones(len(X))] @ Wt


def cell_features(tok, pos_onehot):
    """tok [N, 81, d] -> rows for the 63 map cells: [N*63, d + 63]."""
    N = tok.shape[0]
    X = tok[:, MAP].reshape(N * 63, -1)
    return np.c_[X, np.tile(pos_onehot, (N, 1))]


class Probes:
    def __init__(self, tok, lab, fit):
        po = np.eye(63, dtype=np.float32)
        Xf = cell_features(tok[fit], po)
        mob = lab["mob"][fit][:, :3].transpose(0, 2, 3, 1).reshape(-1, 3).astype(np.float32)   # [N*63, 3]
        self.W_mob = ridge(Xf, mob)
        view = lab["view"][fit].reshape(-1).astype(int)
        self.W_ter = ridge(Xf, np.eye(17, dtype=np.float32)[view])
        Xh = tok[fit][:, HUD].reshape(fit.sum(), -1)
        self.W_vit = ridge(Xh, lab["vit"][fit].astype(np.float32))
        self.po = po

    def score(self, tok, lab_mob, lab_view, lab_vit):
        N = tok.shape[0]
        X = cell_features(tok, self.po)
        pm = apply(self.W_mob, X)
        y = lab_mob[:, :3].transpose(0, 2, 3, 1).reshape(-1, 3)
        ap = {n: ap_grouped(y[:, i], pm[:, i]) for i, n in enumerate(("zombie", "cow", "skeleton"))}
        pt = apply(self.W_ter, X).argmax(1); vt = lab_view.reshape(-1)
        ter = {"all": float((pt == vt).mean()), "non_grass": float((pt[vt != 2] == vt[vt != 2]).mean())}
        pv = apply(self.W_vit, tok[:, HUD].reshape(N, -1)); yv = lab_vit.astype(np.float64)
        r2 = {n: float(1 - ((pv[:, i] - yv[:, i]) ** 2).sum() / max(((yv[:, i] - yv[:, i].mean()) ** 2).sum(), 1e-9))
              for i, n in enumerate(("health", "food", "drink", "energy"))}
        return {"mob_ap": ap, "terrain_acc": ter, "vitals_r2": r2}


def eff_rank(tok):
    out = []
    for q in range(81):
        x = tok[:, q] - tok[:, q].mean(0)
        ev = np.clip(np.linalg.eigvalsh(x.T @ x / len(x)), 0, None)
        p = ev / ev.sum()
        out.append(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())))
    return float(np.mean(out))


def load(d):
    fr = np.load(os.path.join(d, "dev_frames.npy"), mmap_mode="r")
    m = np.load(os.path.join(d, "dev_meta.npz"))
    lab = {"mob": m["lab_mob"], "view": m["lab_view"], "vit": m["lab_vit"]}
    ep = np.zeros(len(fr), int)
    for e, (s, L) in enumerate(zip(m["starts"], m["lengths"])):
        ep[s:s + L] = e
    fit = ep % 2 == 0
    rng = np.random.default_rng(20261028)
    cand = np.concatenate([np.arange(s, s + L - 20 + 1) for e, (s, L) in enumerate(zip(m["starts"], m["lengths"]))
                           if e % 2 == 1 and L >= 20])
    win = np.sort(rng.choice(cand, 512, replace=False))
    return fr, m, lab, fit, win


@torch.no_grad()
def encode(enc, fr, dev="cuda", bs=1024):
    out = []
    for i in range(0, len(fr), bs):
        x = torch.from_numpy(np.ascontiguousarray(fr[i:i + bs])).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x)
        out.append(F.layer_norm(z.float(), (D_LAT,)).cpu().numpy().astype(np.float32))
    return np.concatenate(out)


@torch.no_grad()
def imagined(pred, tok, m, win, dev="cuda"):
    idx = win[:, None] + np.arange(20)[None]
    z = torch.from_numpy(tok[idx]).to(dev)                           # [512, 20, 81, 64] (LN'd real tokens)
    act = torch.from_numpy(m["actions"][idx[:, :-1]].astype(np.int64)).to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        zhat, rhat, chat = pred.unroll(z, act, n_teacher=C)
    zhat = F.layer_norm(zhat.float(), (D_LAT,)).cpu().numpy()        # predictions of frames 1..19
    return zhat, rhat.float().cpu().numpy(), chat.float().cpu().numpy(), idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--run", required=True)
    p.add_argument("--ckpts", default="0,1000,3000,6000,10000"); p.add_argument("--raw", action="store_true")
    a = p.parse_args()
    fr, m, lab, fit, win = load(a.data)
    sc = ~fit
    res = {}
    if a.raw:                                                         # raw-pixel reference
        P = Encoder.patches(torch.from_numpy(np.ascontiguousarray(fr[:]))).numpy().astype(np.float32)
        pr = Probes(P, lab, fit)
        res["raw_pixels"] = pr.score(P[sc], lab["mob"][sc], lab["view"][sc], lab["vit"][sc])
        json.dump(res, open(os.path.join(a.run, "eval_raw.json"), "w"), indent=1); print(json.dumps(res)); return
    for ck in [int(x) for x in a.ckpts.split(",")]:
        path = os.path.join(a.run, f"ckpt_{ck:05d}.pt")
        if not os.path.exists(path):
            continue
        st = torch.load(path, weights_only=False)
        enc, pred = Encoder().cuda().eval(), Predictor().cuda().eval()
        enc.load_state_dict(st["enc"]); pred.load_state_dict(st["pred"])
        tok = encode(enc, fr)
        pr = Probes(tok, lab, fit)
        r = pr.score(tok[sc], lab["mob"][sc], lab["view"][sc], lab["vit"][sc])
        r["eff_rank"] = eff_rank(tok[sc][::4])
        zhat, rhat, chat, idx = imagined(pred, tok, m, win)
        depth = {}
        for k in (1, 2, 4, 8, 12, 16):
            j = 3 + k                                                  # frame index in the window
            fi = idx[:, j]
            im = pr.score(zhat[:, j - 1], lab["mob"][fi], lab["view"][fi], lab["vit"][fi])
            cp = pr.score(tok[idx[:, 3]], lab["mob"][fi], lab["view"][fi], lab["vit"][fi])
            ce = pr.score(tok[fi], lab["mob"][fi], lab["view"][fi], lab["vit"][fi])
            num = ((zhat[:, j - 1] - tok[fi]) ** 2).sum(); den = ((tok[idx[:, 3]] - tok[fi]) ** 2).sum()
            depth[k] = {"imagined": im, "copy": cp, "ceiling": ce, "rel_latent_err": float(num / max(den, 1e-9))}
        rr = m["rewards"][idx[:, :-1]]
        depth["reward_mse"] = float(((rhat - rr) ** 2).mean()); depth["reward_mse_zero"] = float((rr ** 2).mean())
        r["imagined_by_depth"] = depth
        res[ck] = r
        print(ck, json.dumps({"mob_ap": r["mob_ap"], "vitals_r2": r["vitals_r2"], "eff_rank": r["eff_rank"],
                              "d8_zombie": depth[8]["imagined"]["mob_ap"]["zombie"], "d8_copy": depth[8]["copy"]["mob_ap"]["zombie"]}), flush=True)
    json.dump(res, open(os.path.join(a.run, "eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
