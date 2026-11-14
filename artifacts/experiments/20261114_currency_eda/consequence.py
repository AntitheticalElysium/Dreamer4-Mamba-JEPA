"""Reading 1 (PLAN.md): the nats of future reward that each hidden counter's phase explains, by distance. CPU.

    python -m artifacts.experiments.20261114_currency_eda.consequence
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
DIST = (1, 2, 4, 8, 16, 32)
COUNTERS = ("hunger", "thirst", "fatigue")


def reward_class(r):
    return np.select([r <= -0.15, r < -0.05, r < 0.05, r < 0.2], [0, 1, 2, 3], 4)


def main():
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    C = dict(np.load(HERE.parent / "20261107_f4_followups/readings/counters_dev.npz"))
    T = L["ep_T"]
    so, ao = np.r_[0, np.cumsum(T + 1)], np.r_[0, np.cumsum(T)]
    N = so[-1]
    ep = np.repeat(np.arange(len(T)), T + 1)
    t = np.arange(N) - so[ep]
    pos = L["s_pos"].astype(int)
    feats = {"body": L["s_body"].astype(np.float32), "inv": L["s_inv"].astype(np.float32),
             "sleep": L["s_sleep"].astype(np.float32)[:, None], "light": L["s_light"].astype(np.float32)[:, None],
             "ach": L["s_ach"].astype(np.float32), "dir": L["s_dir"].astype(np.float32)[:, None]}
    for k in ("zombies", "cows", "skeletons", "arrows"):
        d = np.abs(L[f"s_{k}_p"].astype(int) - pos[:, None]).sum(-1)
        m = L[f"s_{k}_m"]
        cols = [(m & (d == 1)).sum(1), (m & (d <= 3)).sum(1), (m & (d <= 6)).sum(1)]
        if k == "zombies":
            cd = np.where(m & (d == 1), L["s_zombies_cd"], 99).min(1)
            cols.append(cd)
        feats[k] = np.stack(cols, 1).astype(np.float32)
    visible = np.concatenate(list(feats.values()), 1)
    cnt = {k: C[k].astype(np.float32)[:, None] for k in (*COUNTERS, "recover")}
    base = np.concatenate([visible, cnt["recover"]], 1)
    variants = {"visible_only": visible, "base": base, **{f"base+{k}": np.concatenate([base, cnt[k]], 1) for k in COUNTERS},
                "base+all": np.concatenate([base, *[cnt[k] for k in COUNTERS]], 1)}
    fold = ep % 5
    fit, val, test = fold <= 2, fold == 3, fold == 4
    body = L["s_body"].astype(int)
    strata = {"all": np.ones(N, bool), "food<=1": body[:, 1] <= 1, "drink<=1": body[:, 2] <= 1, "energy<=1": body[:, 3] <= 1,
              "asleep": L["s_sleep"].astype(bool)}
    rng = np.random.default_rng(0)
    te = np.flatnonzero(test)
    eps_te, inv = np.unique(ep[te], return_inverse=True)
    boots = rng.integers(0, len(eps_te), (1000, len(eps_te)))

    def gain(d, mask):
        s, n = np.zeros(len(eps_te)), np.zeros(len(eps_te))
        np.add.at(s, inv[mask], d[mask]); np.add.at(n, inv[mask], 1)
        b = s[boots].sum(1) / np.maximum(n[boots].sum(1), 1)
        return [float(s.sum() / max(n.sum(), 1)), *np.quantile(b, [.025, .975]).tolist(), int(mask.sum())]

    res = {"states": int(N), "episodes": int(len(T)), "test_states": int(test.sum()), "distances": {}}
    reward = L["t_reward"]
    for n_ in DIST:
        y = np.full(N, 5)
        ok = t + n_ <= T[ep]
        y[ok] = reward_class(reward[ao[ep[ok]] + t[ok] + n_ - 1])
        ll = {}
        for name, X in variants.items():
            params = dict(objective="multiclass", num_class=6, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                          bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, num_threads=6, verbose=-1, seed=0)
            m = lgb.train(params, lgb.Dataset(X[fit], y[fit]), num_boost_round=600,
                          valid_sets=[lgb.Dataset(X[val], y[val])], callbacks=[lgb.early_stopping(30, verbose=False)])
            p = m.predict(X[te], num_iteration=m.best_iteration)
            ll[name] = -np.log(np.clip(p[np.arange(len(te)), y[te]], 1e-9, 1))
        out = {"class_shares": np.bincount(y[te], minlength=6).tolist(), "log_loss": {k: float(v.mean()) for k, v in ll.items()},
               "recover_gain_over_visible": gain(ll["visible_only"] - ll["base"], np.ones(len(te), bool)), "gains": {}}
        for k in COUNTERS:
            out["gains"][k] = {sn: gain(ll["base"] - ll[f"base+{k}"], sm[te]) for sn, sm in strata.items()}
        out["gains"]["all_three"] = {"all": gain(ll["base"] - ll["base+all"], np.ones(len(te), bool))}
        res["distances"][n_] = out
        print(n_, json.dumps(out), flush=True)
    (HERE / "consequence.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
