import sys, collections, numpy as np
sys.path.insert(0, "artifacts/eda"); import replay as RP
from saccade.eval import gates as G
L = dict(np.load("artifacts/eda/s0/s0_dev_labels.npz")); eps = G.episode_grids(L)
T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]
H = lambda c: -sum(v / sum(c.values()) * np.log2(v / sum(c.values())) for v in c.values())
by_sem = collections.defaultdict(list); exact1 = collections.Counter(); j1 = collections.Counter(); sem1 = collections.Counter()
rng = np.random.default_rng(0)
for e in rng.permutation(len(T))[:300]:
    obs = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"].numpy()
    ep = eps[e]; light = L["s_light"][s_off[e]:s_off[e + 1]]; d = L["s_dir"][s_off[e]:s_off[e + 1]]
    for t in range(0, len(obs), 3):
        if light[t] < 0.99 or L["s_sleep"][s_off[e] + t]: continue
        cells = obs[t].reshape(9, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4)
        for i in range(7):
            for j in range(9):
                s = (int(ep["view"][t, i, j]), int(ep["grid"][t, i, j])) + ((int(d[t]),) if (i, j) == (3, 4) else ())
                if len(by_sem[s]) < 400: by_sem[s].append(cells[i, j].astype(np.int16))
                if light[t] >= 0.99999:
                    h = cells[i, j].tobytes(); exact1[h] += 1; j1[(s, h)] += 1; sem1[s] += 1
print("light == 1.0 exactly: H(pixels | semantics) =", round(H(j1) - H(sem1), 3), "bits over", sum(sem1.values()), "cells")
rows = []
for s, P in by_sem.items():
    P = np.stack(P); m = np.median(P, 0)
    dev = np.abs(P - m).mean((1, 2, 3)); n_distinct = len({p.tobytes() for p in P})
    rows.append((s, len(P), n_distinct, float(np.median(dev)), float(np.percentile(dev, 99)), float(dev.max())))
rows.sort(key=lambda r: -r[4])
print("semantic, n, distinct patterns, within-meaning pixel deviation (mean abs): median / p99 / max  [top 12 by p99]")
for r in rows[:12]: print("  ", r[0], r[1], r[2], round(r[3], 1), round(r[4], 1), round(r[5], 1))
