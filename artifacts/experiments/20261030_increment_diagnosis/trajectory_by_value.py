"""Part 3 of PLAN.md: drink and wood generation by start value over P1 world 1's checkpoints. The probe's rows (eye
elsewhere at the target frame; fixations do not depend on the checkpoint), the probe's draws (one generator per batch,
seeded by the batch), stored beliefs; each checkpoint encoded and read with its own encoder's value means, whose
instrument check is the read of the real end tokens of the same rows.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_increment_diagnosis.trajectory_by_value OUT_JSON
  [N_EPISODES: smoke tests only]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M
from saccade.eval import s1 as S, gates as G

IP = importlib.import_module("artifacts.experiments.20261030_increment_diagnosis.increment_probe")
EV, DS, CAT = IP.EV, IP.DS, IP.CAT
RUN = EV.ROOT / "artifacts/saccade_m0/p1_world1"
CKPTS = (1, 2, 3, 5, 7, 10)


def main():
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    L = S.first_episodes(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rows = DS.rows_from_labels(L)
    res = {"checkpoints": {}}
    for k in CKPTS:
        ck = torch.load(RUN / f"ckpt_{k:02d}.pt", weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(IP.DEV); model.load_state_dict(ck["model"]); model.eval()
        frames = int(ck["state"]["frames"]); del ck
        enc = S.encode(model, L, cfg)
        stats = IP.value_stats(L, enc)
        R_ = {"frames": frames}
        for kind in IP.KINDS:
            st = stats[kind]
            cell, arr, col = CAT.HUD[kind]
            x = L["s_body" if arr == 0 else "s_inv"][:, col]
            ctxs = [(e, t, int(x[s_off[e] + t - 1])) for e, t, a in rows[kind] if not (M.window(enc[e]["fix"][t]) == cell).any()]
            reads, fields, real = [], [], []
            for b0 in range(0, len(ctxs), 256 // IP.R):
                batch = ctxs[b0:b0 + 256 // IP.R]
                inps = [i for c in batch for i in [IP.row_inputs(model, cfg, enc[c[0]], c[1], {}, st)] * IP.R]
                fd, xg = IP.generate(model, cfg, inps, IP.SEED + b0)
                v = np.repeat([c[2] for c in batch], IP.R)
                reads.append(IP.read(st, xg[:, cell])[0])
                a_ = st["means"][v]; w_ = st["means"][np.minimum(v + 1, 9)] - a_
                f = ((fd[:, cell] - a_) * w_).sum(-1) / w_.square().sum(-1)
                fields.append(torch.where(w_.square().sum(-1) >= CAT.MIN_SEPARATION, f, torch.full_like(f, float("nan"))).numpy())
                real.append(IP.read(st, torch.stack([enc[c[0]]["p"][c[1], cell].float() for c in batch]))[0] == np.array([c[2] + 1 for c in batch]))
            r, fl, rl = np.concatenate(reads), np.concatenate(fields), np.concatenate(real)
            v = np.repeat([c[2] for c in ctxs], IP.R); ep = np.repeat([c[0] for c in ctxs], IP.R)
            est = lambda val, m: (lambda z: None if z is None else {"est": round(z["est"], 4), "se": round(z["se"], 4), "n": int(m.sum())})(
                G.estimate(np.where(m, val, 0.0), m, ep) if m.any() else None)
            grp = {"appearance": v == 0, "increment": v >= 1, **{f"v={u}": v == u for u in np.unique(v)}}
            R_[kind] = {"real_end_read_accuracy": float(rl.mean()), "contexts": len(ctxs),
                        **{g: {"success": est((r == v + 1).astype(float), m), "copy": est((r == v).astype(float), m),
                               "field_20": est(np.nan_to_num(fl), m & np.isfinite(fl))} for g, m in grp.items()}}
            print(k, kind, "real", round(float(rl.mean()), 3), {g: (R_[kind][g]["success"] or {}).get("est") for g in grp}, flush=True)
        res["checkpoints"][k] = R_
        out.write_text(json.dumps(res, indent=1))
        del model, enc
        torch.cuda.empty_cache()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
