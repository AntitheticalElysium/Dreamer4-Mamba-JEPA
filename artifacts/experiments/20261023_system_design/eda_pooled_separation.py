"""EDA: separation of distinct meanings after pooling a cell to r x r (mean colour per block): min / median pairwise
RMS distance (per pooled value, 0-255 scale) between prototypes of different meaning. Map: (terrain, entity) at full
daylight, not sleeping. HUD: each displayed value per position (HUD has no static)."""
import sys, collections, numpy as np
sys.path.insert(0, "artifacts/eda"); import replay as RP
from saccade.eval import gates as G
L = dict(np.load("artifacts/eda/s0/s0_dev_labels.npz")); eps = G.episode_grids(L)
T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]
proto_map, proto_hud = {}, collections.defaultdict(dict)
rng = np.random.default_rng(0)
for e in rng.permutation(len(T))[:300]:
    obs = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"].numpy(); ep = eps[e]
    light = L["s_light"][s_off[e]:s_off[e + 1]]; sl = L["s_sleep"][s_off[e]:s_off[e + 1]]
    for t in range(0, len(obs), 5):
        cells = obs[t].reshape(9, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4).astype(np.float32)
        if light[t] >= 0.99999 and not sl[t]:
            for i in range(7):
                for j in range(9):
                    if (i, j) != (3, 4): proto_map.setdefault((int(ep["view"][t, i, j]), int(ep["grid"][t, i, j])), cells[i, j])
        for j in (0, 1, 2, 3, 5, 6, 7, 8):
            proto_hud[j].setdefault(cells[7, j].tobytes(), cells[7, j])
def pool(c, r):
    e = np.linspace(0, 7, r + 1).round().astype(int)
    return np.stack([np.stack([c[e[a]:e[a + 1], e[b]:e[b + 1]].mean((0, 1)) for b in range(r)]) for a in range(r)]).ravel()
def sep(P, r):
    X = np.stack([pool(p, r) for p in P]); D = np.sqrt(((X[:, None] - X[None]) ** 2).mean(-1)); iu = np.triu_indices(len(X), 1)
    return D[iu].min(), np.median(D[iu])
for r in (7, 3, 2, 1):
    m = sep(list(proto_map.values()), r)
    h = [sep(list(v.values()), r) for v in proto_hud.values() if len(v) > 2]
    print(f"pool {r}x{r}: map meanings ({len(proto_map)}) min {m[0]:.1f} median {m[1]:.1f} | "
          f"HUD values (8 positions) min over positions {min(x[0] for x in h):.1f}, median of mins {np.median([x[0] for x in h]):.1f}")
