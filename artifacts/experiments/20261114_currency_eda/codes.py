"""Reading 2 (PLAN.md): a learned discrete code over F6's periphery tokens: what changes it, and what it would price.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_currency_eda.codes
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
DZ = importlib.import_module("artifacts.experiments.20261114_f6_dissection.dose")
G2, TJ, DEV = DZ.G2, DZ.TJ, DZ.DEV
KS, FOILS = (64, 256, 1024), (1, 3, 6)
DIGIT = {"health": 63, "food": 64, "drink": 65, "energy": 66}


def kmeans(x, k, iters=25, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randperm(len(x), device=x.device, generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cat([torch.cdist(x[i:i + 262144], c).argmin(1) for i in range(0, len(x), 262144)])
        s = torch.zeros_like(c).index_add_(0, a, x)
        n = torch.bincount(a, minlength=k).clamp(min=1)[:, None]
        c = torch.where(torch.bincount(a, minlength=k)[:, None] > 0, s / n, c)
    return c


def assign(x, c):
    flat = x.reshape(-1, x.shape[-1])
    return torch.cat([torch.cdist(flat[i:i + 262144].float(), c).argmin(1) for i in range(0, len(flat), 262144)]).reshape(x.shape[:-1])


@torch.no_grad()
def main():
    ckp, model, cfg, _ = DZ.load(10)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    enc = [r["p"] for r in S.encode(model, L, cfg)]                       # [T + 1, 81, 16] half, cpu
    fit = [e for e in range(len(enc)) if e % 5 <= 2]
    test = [e for e in range(len(enc)) if e % 5 == 4]
    g = torch.Generator().manual_seed(0)
    pool = torch.cat([enc[e].reshape(-1, 16) for e in fit])
    pool = pool[torch.randperm(len(pool), generator=g)[:3_000_000]].float().to(DEV)
    books = {k: kmeans(pool, k) for k in KS}
    del pool
    eps = S.episodes(L)
    res = {"checkpoint_sha256": TR.file_sha256(ckp), "K": {}}
    rng = np.random.default_rng(0)
    for k, c in books.items():
        acc, ev, rate_n, rate_d = {}, {n: [0, 0] for n in DIGIT}, np.zeros((2, 81)), np.zeros((2, 81))
        for e in test:
            code = assign(enc[e].to(DEV), c).cpu().numpy()                # [T + 1, 81]
            ch = code[1:] != code[:-1]
            a, b = off[e], off[e + 1]
            pos, sleep, lt, body = L["s_pos"][a:b].astype(int), L["s_sleep"][a:b].astype(bool), L["s_light"][a:b], L["s_body"][a:b].astype(int)
            moved = (pos[1:] != pos[:-1]).any(-1)
            cls = np.where(sleep[:-1], "asleep", np.where(moved, "moved", np.where(lt[1:] >= .99, "still_day", np.where(lt[1:] >= .5, "still_dusk_dawn", "still_night"))))
            dch = body[1:] != body[:-1]                                    # [T, 4]
            night = (lt[1:] < .5).astype(int)
            np.add.at(rate_n, night, ch); np.add.at(rate_d, night, 1)
            quiet_digit = ch[:, 63:67][~dch]
            for name in np.unique(cls):
                m = cls == name
                z = acc.setdefault(name, np.zeros(6))
                z += [ch[m][:, :63].sum(), ch[m][:, :63].size, ch[m][:, 63:67][~dch[m]].sum(), (~dch[m]).sum(), ch[m][:, 67:].sum(), ch[m][:, 67:].size]
            for j, n in enumerate(DIGIT):
                ev[n][0] += int(ch[:, 63 + j][dch[:, j]].sum()); ev[n][1] += int(dch[:, j].sum())
        rate = np.clip(rate_n / np.maximum(rate_d, 1), 1e-5, 1)           # each cell's own change rate, by day / night
        out = {"events_changing_their_code": {n: v[0] / max(v[1], 1) for n, v in ev.items()},
               "code_change_rate": {n: {"map": z[0] / z[1], "hud_digits_no_digit_change": z[2] / max(z[3], 1), "hud_other": z[4] / z[5]} for n, z in acc.items()},
               "digit_cell_change_rate_day_night": {n: rate[:, cell].tolist() for n, cell in DIGIT.items()}}
        # brightness on test frames
        fl = {a: [0, 0] for a in FOILS}
        for e in rng.choice(test, 40, replace=False):
            frames = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy()
            ts = rng.choice(len(frames), min(32, len(frames)), replace=False)
            f = torch.as_tensor(frames[ts], device=DEV)
            fx = torch.full((len(ts), 2), 4, device=DEV, dtype=torch.long)
            base = assign(model.perception(f, fx).p.float(), c)
            for a in FOILS:
                alt = assign(model.perception(torch.clamp(f.int() + a, max=255).to(torch.uint8), fx).p.float(), c)
                fl[a][0] += int((alt != base).sum()); fl[a][1] += base.numel()
        out["cells_changing_code_under_brightness"] = {str(a): v[0] / v[1] for a, v in fl.items()}
        # prices on the exact-history pairs
        out["pairs"] = {}
        for name, slot in (("energy", 66), ("hunger", 64)):
            rows_ = TJ.pair_rows(name)
            cons, seen, foil = [], [], {a: [] for a in FOILS}
            for row in rows_:
                x, xb = G2.due_targets(model, row)
                i, e_, prefix, frames, rewards, terms, q, truth = row
                night = 0                                                  # day rates: the pair banks start from daytime states
                cx = assign(x, c).cpu().numpy()
                diff = cx[0] != cx[1]
                seen.append(bool(diff[slot]))
                cons.append(float(-np.log(rate[night, slot])) if diff[slot] else 0.)
                for a in FOILS:
                    d = assign(xb[a], c).cpu().numpy() != cx[1]
                    foil[a].append(float((-np.log(rate[night])[d]).sum()))
            out["pairs"][name] = {"pairs": len(rows_), "due_member_changes_the_digit_code": int(sum(seen)),
                                  "consequence_nats_median": float(np.median(cons)),
                                  "foil_nats_median": {str(a): float(np.median(v)) for a, v in foil.items()},
                                  "foil_beats_consequence": {str(a): int(sum(f_ >= c_ for f_, c_ in zip(v, cons))) for a, v in foil.items()}}
        res["K"][k] = out
        print(k, json.dumps(out), flush=True)
    (HERE / "codes.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main()
