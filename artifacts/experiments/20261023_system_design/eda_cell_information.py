"""EDA (facts about Craftax frames, not an architecture test): information per 7x7 cell.
Semantic content: map cells (terrain class, entity kind; player cell adds facing); HUD cells = their exact pixels
(the HUD is not darkened or noised at night). Appearance beyond semantics: exact-pixel entropy by full daylight minus
semantic entropy. Detail scale: mean L1 pixel distance between cells of different meaning, per region."""
import sys, collections, numpy as np
sys.path.insert(0, "artifacts/eda"); import replay as RP
from saccade.eval import gates as G
L = dict(np.load("artifacts/eda/s0/s0_dev_labels.npz")); eps = G.episode_grids(L)
T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]
H = lambda c: -sum(v / sum(c.values()) * np.log2(v / sum(c.values())) for v in c.values())
sem_map, app_day, joint_day = collections.Counter(), collections.Counter(), collections.Counter()
sem_ring = collections.defaultdict(collections.Counter)
hud = collections.defaultdict(collections.Counter); proto = {}
RING = np.maximum(np.abs(np.arange(7)[:, None] - 3), np.abs(np.arange(9)[None] - 4))
rng = np.random.default_rng(0); light_all = []
for e in rng.permutation(len(T))[:300]:
    obs = RP.episode_fields(int(L["ep_shard"][e]), int(L["ep_slot"][e]))["observations"].numpy()
    ep = eps[e]; light = L["s_light"][s_off[e]:s_off[e + 1]]; d = L["s_dir"][s_off[e]:s_off[e + 1]]
    for t in range(0, len(obs), 3):
        cells = obs[t].reshape(9, 7, 9, 7, 3).transpose(0, 2, 1, 3, 4)          # [9 rows, 9 cols, 7, 7, 3]
        light_all.append(light[t])
        for i in range(7):
            for j in range(9):
                s = (int(ep["view"][t, i, j]), int(ep["grid"][t, i, j])) + ((int(d[t]),) if (i, j) == (3, 4) else ())
                sem_map[s] += 1; sem_ring[RING[i, j]][s] += 1
                if light[t] >= 0.99:
                    h = cells[i, j].tobytes(); app_day[h] += 1; joint_day[(s, h)] += 1
                    proto.setdefault(("map", s), cells[i, j].astype(np.float32))
        for i in (7, 8):
            for j in range(9):
                h = cells[i, j].tobytes(); hud[(i, j)][h] += 1; proto.setdefault(("hud", (i, j), h), cells[i, j].astype(np.float32))
light_all = np.array(light_all)
print(f"frames {len(light_all)}, full daylight (>=0.99) {np.mean(light_all >= 0.99):.2f}")
Hs, Ha = H(sem_map), H(app_day)
sem_day = collections.Counter(); [sem_day.__setitem__(s, sem_day[s] + n) for (s, h), n in joint_day.items()]
print(f"MAP cell semantics: {len(sem_map)} distinct (terrain, entity[, facing]); entropy {Hs:.2f} bits")
print("   by ring:", {r: round(H(c), 2) for r, c in sorted(sem_ring.items())})
print(f"MAP cell exact pixels, full daylight: {len(app_day)} distinct; entropy {Ha:.2f} bits; "
      f"H(pixels | semantics) = {H(joint_day) - H(sem_day):.3f} bits; H(semantics | pixels) = {H(joint_day) - Ha:.3f} bits")
tot = 0
for k, c in sorted(hud.items()):
    tot += H(c)
print(f"HUD cells: entropy per cell (exact pixels, = meaning): " + ", ".join(f"{k[0]-7},{k[1]}:{H(c):.2f}" for k, c in sorted(hud.items())))
print(f"   mean {tot / len(hud):.2f} bits per HUD cell; distinct patterns per cell: {[len(c) for k, c in sorted(hud.items())]}")
# detail scale: L1 distance (mean abs per pixel channel) between distinct meanings
def dists(keys):
    P = np.stack([proto[k] for k in keys]); D = np.abs(P[:, None] - P[None]).mean((2, 3, 4)); iu = np.triu_indices(len(P), 1)
    return D[iu]
mk = [k for k in proto if k[0] == "map" and len(k[1]) == 2 and k[1][1] == 0]          # terrain classes, no entity
d = dists(mk); print(f"pixel L1 between terrain classes (day): median {np.median(d):.1f}, min {d.min():.1f} (n={len(mk)} classes)")
ek = [k for k in proto if k[0] == "map" and len(k[1]) == 2 and k[1][0] == 2]           # grass with each entity kind
d = dists(ek); print(f"pixel L1 between entity kinds on grass (day): median {np.median(d):.1f}, min {d.min():.1f} (n={len(ek)})")
for pos in [(7, 0), (7, 5)]:
    hk = [k for k in proto if k[0] == "hud" and k[1] == pos]
    if len(hk) > 1:
        d = dists(hk); print(f"pixel L1 between HUD values at {pos[0]-7},{pos[1]}: median {np.median(d):.1f}, min {d.min():.1f} (n={len(hk)})")
