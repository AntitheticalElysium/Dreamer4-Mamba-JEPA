"""S0 step 6 (2026-10-27): reward and continuation evaluator, with information references (TEST_SPEC v1.1, S1.3).

The outcome models predict reward from (predicted belief, next percept) and continuation from (predicted belief, next
percept, reward) (STEP_SPEC 2.5). In Craftax-Classic (dev labels): reward = unlocks + 0.1 * health change, the unlock
part is 0 or 1 per step; termination happens exactly when the HUD shows health 0 at t+1 (no other case).
So the health part and continuation are READABLE from the next percept; the unlock part is not displayed and needs the
visible changes plus memory of which achievements are already unlocked.
Evaluator (applied later to model outputs): reward NLL in nats of the recorded value (each recorded value is its own
bin of the discretised density), unlock-event Brier and ECE (10 equal-mass bins), continuation NLL / Brier / ECE;
episode-bootstrap SE for the means.
Common outcome bins (LOSSES G3): 256 uniform bins in symlog space over REWARD_SUPPORT. A learned reward density is
scored by its mass on the recorded reward's bin (integrate the density over the bin, never compare density values);
the categorical references below put mass on reward VALUES, which equals bin mass because every observed value has its
own bin (checked in main).
Information references (fit on train-400 minus 40 held-out episodes for early stopping, read on all dev). They are
FINITE-LEARNER references (LightGBM on chosen features), not information limits: R2 is not a memory-free bound
(review 10-27: adding visible entity / terrain counts of the same frame pair, no older history, lowered its NLL by
13%), and R2 / R3 read the simulator's neighbouring terrain even under an entity, which a pixel reader cannot see.
Memory is established by the matched controls and direct probes of TEST_SPEC S1.2, not by beating R2.
  R0 marginal reward / continuation frequencies;
  R1 + the HUD health part (exact) with the marginal unlock rate;
  R2 + visible changes of the percept pair (HUD and inventory at t and t+1, sleep, light, action, facing direction,
       the 4 neighbouring terrain cells and adjacent entity kinds at t and t+1), LightGBM, no memory;
  R3 = R2 + the achievements unlocked before t (perfect memory).
Usage (scratch venv with numpy + lightgbm): python -m saccade.eval.calib TRAIN_LABELS_NPZ DEV_LABELS_NPZ OUT_JSON
"""
import json, sys
import numpy as np
from saccade.eval import gates as G

NB = ((2, 4), (4, 4), (3, 3), (3, 5))                               # the 4 neighbours of the player in the 7x9 view
REWARD_SUPPORT = (-0.9, 1.1)          # the support-v2 train split's reward range: 21 values, -0.9 .. 1.1 in 0.1 steps


def reward_bin(r, n=256):
    """Index of r's LOSSES G3 bin: n uniform bins in symlog space over REWARD_SUPPORT (edges clipped)."""
    lo, hi = (np.sign(x) * np.log1p(abs(x)) for x in REWARD_SUPPORT)
    x = np.sign(r) * np.log1p(np.abs(r))
    return np.clip(np.floor((x - lo) / (hi - lo) * n).astype(int), 0, n - 1)


def transitions(L):
    T = L["ep_T"]; s_off = np.r_[0, np.cumsum(T + 1)]
    i0 = np.concatenate([np.arange(s_off[e], s_off[e] + T[e]) for e in range(len(T))]); i1 = i0 + 1
    grid = np.concatenate([ep["grid"] for ep in G.episode_grids(L)])
    body = L["s_body"].astype(float)
    dh = body[i1, 0] - body[i0, 0]
    unlock = np.round(L["t_reward"] - 0.1 * dh).astype(int)
    pair = [body[i0], body[i1], L["s_inv"][i0], L["s_inv"][i1], L["s_sleep"][i0, None], L["s_sleep"][i1, None],
            L["s_light"][i1, None], L["t_action"][:, None], L["s_dir"][i0, None],
            np.stack([L["s_view"][i0, r, c] for r, c in NB] + [L["s_view"][i1, r, c] for r, c in NB], 1),
            np.stack([grid[i0, r, c] for r, c in NB] + [grid[i1, r, c] for r, c in NB], 1)]
    X2 = np.concatenate([np.asarray(x, float).reshape(len(i0), -1) for x in pair], 1)
    X3 = np.concatenate([X2, L["s_ach"][i0].astype(float)], 1)
    return {"X2": X2, "X3": X3, "unlock": unlock, "reward": L["t_reward"], "term": L["t_terminated"],
            "health1": body[i1, 0], "ep": np.repeat(np.arange(len(T)), T)}


def ece(p, y, bins=10):
    """Expected calibration error with equal-mass bins."""
    o = np.argsort(p); parts = np.array_split(o, bins)
    return float(sum(len(b) * abs(p[b].mean() - y[b].mean()) for b in parts if len(b)) / len(p))


def score(p_true, p_event, y_event, ep):
    """p_true: probability of the recorded outcome; p_event / y_event: event probability and indicator."""
    nll = -np.log(np.maximum(p_true, 1e-12)); one = np.ones(len(ep), bool)
    return {"nll": G.estimate(nll, one, ep), "brier": G.estimate((p_event - y_event) ** 2, one, ep),
            "ece": ece(p_event, y_event.astype(float))}


def main():
    import lightgbm as lgb
    tr, dv = transitions(dict(np.load(sys.argv[1]))), transitions(dict(np.load(sys.argv[2])))
    held = tr["ep"] % 10 == 9
    out = {"dev_unlock_rate": float(dv["unlock"].mean()), "dev_termination_rate": float(dv["term"].mean()),
           "termination_iff_health0": bool(((dv["health1"] == 0) == dv["term"]).all()), "reward": {}, "continuation": {}}
    key = lambda r: np.round(r * 10).astype(int)
    vals, cnt = np.unique(key(tr["reward"]), return_counts=True); freq = dict(zip(vals, cnt / cnt.sum()))
    p_u = tr["unlock"].mean(); y = dv["unlock"].astype(float)
    p_true = np.array([freq.get(k, 0.0) for k in key(dv["reward"])])
    out["reward"]["R0_marginal"] = score(p_true, np.full(len(y), p_u), y, dv["ep"])
    out["reward"]["R0_marginal"]["dev_values_unseen_in_train"] = int((p_true == 0).sum())
    out["reward"]["R1_hud_health"] = score(np.where(y == 1, p_u, 1 - p_u), np.full(len(y), p_u), y, dv["ep"])
    params = {"objective": "binary", "verbose": -1, "num_threads": 12, "seed": 0, "num_leaves": 31, "learning_rate": 0.05,
              "lambda_l2": 1.0, "min_sum_hessian_in_leaf": 1.0}
    for name, X in (("R2_pair_no_memory", "X2"), ("R3_pair_memory", "X3")):
        m = lgb.train(params, lgb.Dataset(tr[X][~held], tr["unlock"][~held]), 2000,
                      valid_sets=[lgb.Dataset(tr[X][held], tr["unlock"][held])], callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(dv[X], num_iteration=m.best_iteration)
        out["reward"][name] = score(np.where(y == 1, p, 1 - p), p, y, dv["ep"]) | {"rounds": int(m.best_iteration)}
    c = dv["term"].astype(float); p_c = tr["term"].mean()
    out["continuation"]["R0_marginal"] = score(np.where(c == 1, p_c, 1 - p_c), np.full(len(c), p_c), c, dv["ep"])
    out["continuation"]["R2_hud_health0"] = score(np.ones(len(c)), (dv["health1"] == 0).astype(float), c, dv["ep"])
    vals = np.unique(np.round(np.r_[tr["reward"], dv["reward"]], 4))
    out["reward_bins"] = {"n": 256, "support": list(REWARD_SUPPORT), "distinct_values": int(len(vals)),
                          "values_sharing_a_bin": int(len(vals) - len(np.unique(reward_bin(vals)))),
                          "values_outside_support": int(((vals < REWARD_SUPPORT[0]) | (vals > REWARD_SUPPORT[1])).sum())}
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
