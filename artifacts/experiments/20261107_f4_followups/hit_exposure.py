"""Danger leads, exposure (CPU, labels only): zombie hits in the train split by situation at t - 1 (ready = an adjacent
zombie with cooldown <= 0; fresh = none adjacent at t - 2), the health drop and its size. Writes readings/hit_exposure.json."""
import numpy as np, json
L = np.load('artifacts/eda/s0/s0_train_labels.npz')
ep_T = L['ep_T']; s_off, a_off = np.r_[0, np.cumsum(ep_T + 1)], np.r_[0, np.cumsum(ep_T)]
ep = np.repeat(np.arange(len(ep_T)), ep_T); g = s_off[ep] + np.arange(len(ep)) - a_off[ep]
b0, b1 = L['s_body'][g], L['s_body'][g + 1]
pos = L['s_pos'][g].astype(int)
zp, zm, zc = L['s_zombies_p'][g].astype(int), L['s_zombies_m'][g], L['s_zombies_cd'][g]
adj = zm & (np.abs(zp - pos[:, None]).sum(-1) == 1)
ready = (adj & (zc <= 0)).any(1); adjacent = adj.any(1)
asleep = L['s_sleep'][g]
drop = b1[:, 0] < b0[:, 0]; delta = b1[:, 0].astype(int) - b0[:, 0].astype(int)
prev = np.maximum(g - 1, 0); first = np.isin(g, s_off)
adj_prev = (L['s_zombies_m'][prev] & (np.abs(L['s_zombies_p'][prev].astype(int) - L['s_pos'][prev].astype(int)[:, None]).sum(-1) == 1)).any(1) & ~first
n = len(g)
out = {'transitions': int(n), 'health_drops': int(drop.sum())}
for name, m in (('awake_ready', ready & ~asleep), ('asleep_ready', ready & asleep), ('asleep_ready_fresh', ready & asleep & ~adj_prev),
                ('awake_ready_fresh', ready & ~asleep & ~adj_prev), ('awake_cooling', adjacent & ~ready & ~asleep), ('not_adjacent', ~adjacent)):
    out[name] = {'rows': int(m.sum()), 'drops': int((m & drop).sum()), 'P(drop)': round(float(drop[m].mean()), 3) if m.any() else None,
                 'deltas': {int(d): int(c) for d, c in zip(*np.unique(delta[m & drop], return_counts=True))}}
open("artifacts/experiments/20261107_f4_followups/readings/hit_exposure.json", "w").write(json.dumps(out, indent=1)); print(json.dumps(out))
