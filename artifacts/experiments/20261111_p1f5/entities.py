"""Where does F5 lose entities in one-step generation? (RESULTS.md, entities.) Frozen readout, no training.

S1's one-step protocol (saccade.eval.s1.one_step: real histories, the system's own sampler, rng seed 0 -> the same
(episode, t) tuples in every arm) on DEV, F4 / HUD-Change / F5, each read by its own S1 entity reader. Every true
entity in view at frame t is classed from the simulator's per-entity slots:
  persisting  the same slot was alive and in view at t - 1 (split: it stayed put / it moved);
  entering    alive at t - 1 but out of view (scroll-in or walked in);
  spawned     not alive at t - 1.
Recall = the generated frame shows that kind at the entity's cell (exact) or within one cell (near). The same reader
applied to the real successor gives the reader's own ceiling. Invented = generated cells of a kind with no true entity of
that kind within one cell, per frame. Day and night are reported separately.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.entities
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade.eval import s1 as S, gates as G
from saccade.eval.probe import KEEP

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection"
DS = importlib.import_module("artifacts.experiments.20261111_p1f5.dissect")
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
N_TUPLES = 20000


def read_kinds(reader, p, v):
    with torch.no_grad():
        return torch.cat([reader(torch.cat([p[i][:, :63].reshape(-1, p.shape[-1]).half().to(DS.DEV),
                                            v[i].repeat_interleave(63, 0).half().to(DS.DEV)], 1))["kind"].argmax(1).reshape(len(i), 63).cpu()
                          for i in torch.arange(len(p)).split(2048)]).numpy()


def truth_entities(L, eps_, ts):
    """Per tuple: list of (kind, cell, category, moved) for true entities in view at t."""
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos = L["s_pos"].astype(int)
    out = []
    for e, t in zip(eps_, ts):
        g = off[e] + t
        rows = []
        for k, name in enumerate(G.KINDS, start=1):
            P, Mk = L[f"s_{name}_p"].astype(int), L[f"s_{name}_m"].astype(bool)
            rel, prel = P[g] - pos[g], P[g - 1] - pos[g - 1]
            vis = Mk[g] & (np.abs(rel[:, 0]) <= 3) & (np.abs(rel[:, 1]) <= 4)
            pvis = Mk[g - 1] & (np.abs(prel[:, 0]) <= 3) & (np.abs(prel[:, 1]) <= 4)
            for j in np.flatnonzero(vis):
                cat = "persisting" if pvis[j] else ("entering" if Mk[g - 1][j] else "spawned")
                moved = bool((P[g][j] != P[g - 1][j]).any()) if Mk[g - 1][j] else None
                rows.append((k, int((rel[j, 0] + 3) * 9 + rel[j, 1] + 4), cat, moved))
        out.append(rows)
    return out


def near(cells_kind, cell, k):
    r, c = divmod(cell, 9)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 7 and 0 <= cc < 9 and cells_kind[rr * 9 + cc] == k:
                return True
    return False


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    res, tuples = {}, None
    for arm in ("f5", "hud", "f4"):
        ck = DS.CKPTS[arm].resolve()
        model, cfg = CV.IF.load(str(ck))
        reader = S.load_reader(ck.parent / "s1/results.reader.pt", ck, DS.DEV)
        enc = S.encode(model, L, cfg)
        eps_, ts, gen = S.one_step(model, enc, cfg, N_TUPLES, np.random.default_rng(0))
        if tuples is None:
            tuples = (eps_.copy(), ts.copy())
            truth = truth_entities(L, eps_, ts)
            night = L["s_light"][off[eps_] + ts] < .5
        assert np.array_equal(tuples[0], eps_) and np.array_equal(tuples[1], ts)
        real_p = torch.stack([enc[e]["p"][t].float() for e, t in zip(eps_, ts)])
        real_v = torch.stack([enc[e]["v"][t].float() for e, t in zip(eps_, ts)])
        kg, kr = read_kinds(reader, gen["p"], gen["v"]), read_kinds(reader, real_p, real_v)
        del model, enc
        torch.cuda.empty_cache()
        stats = {}
        for light, lm in (("day", ~night), ("night", night)):
            for i in np.flatnonzero(lm):
                for k, cell, cat, moved in truth[i]:
                    for key in (f"{light}/{G.KINDS[k - 1]}/{cat}", f"{light}/all/{cat}") + \
                               ((f"{light}/all/persisting_{'moved' if moved else 'still'}",) if cat == "persisting" else ()):
                        s = stats.setdefault(key, [0, 0, 0, 0, 0])
                        s[0] += 1
                        s[1] += int(kg[i, cell] == k); s[2] += int(near(kg[i], cell, k))
                        s[3] += int(kr[i, cell] == k); s[4] += int(near(kr[i], cell, k))
            truth_k = {}
            for i in np.flatnonzero(lm):
                for k in range(1, 5):
                    true_cells = [cell for kk, cell, _, _ in truth[i] if kk == k]
                    gen_cells = [c for c in np.flatnonzero(kg[i] == k) if KEEP[c]]
                    inv = sum(1 for c in gen_cells if not any(abs(c // 9 - t // 9) <= 1 and abs(c % 9 - t % 9) <= 1 for t in true_cells))
                    truth_k.setdefault(G.KINDS[k - 1], [0, 0, 0])
                    truth_k[G.KINDS[k - 1]][0] += inv; truth_k[G.KINDS[k - 1]][1] += len(gen_cells); truth_k[G.KINDS[k - 1]][2] += 1
            stats[f"{light}/invented_per_frame"] = {k: v[0] / v[2] for k, v in truth_k.items()}
            stats[f"{light}/generated_per_frame"] = {k: v[1] / v[2] for k, v in truth_k.items()}
        res[arm] = {key: ({"n": v[0], "gen_exact": v[1] / v[0], "gen_near": v[2] / v[0], "real_exact": v[3] / v[0], "real_near": v[4] / v[0]}
                          if isinstance(v, list) else v) for key, v in stats.items()}
        print(arm, json.dumps({k: (round(v["gen_near"], 3), round(v["real_near"], 3), v["n"]) for k, v in res[arm].items()
                               if isinstance(v, dict) and "/all/" in k}), flush=True)
        print(arm, "invented", json.dumps(res[arm]["day/invented_per_frame"]), json.dumps(res[arm]["night/invented_per_frame"]), flush=True)
    res["tuples"] = N_TUPLES
    (OUT / "entities.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
