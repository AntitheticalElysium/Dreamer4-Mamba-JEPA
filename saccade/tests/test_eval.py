"""Regression checks for the evaluation harness, one per defect found in the 10-27 review (Codex, verified)."""
import numpy as np
from saccade.eval import gates as G


def test_zero_variance_difference_rejects_and_missing_exposure_is_unmeasured():
    ref = {"dir_ok": {"est": 1.0, "se": 0.0, "n_cond": 10}, "same": {"est": 1.0, "se": 0.0, "n_cond": 10},
           "rare": None}
    est = {"dir_ok": {"est": 0.0, "se": 0.0, "n_cond": 10}, "same": {"est": 1.0, "se": 0.0, "n_cond": 10},
           "rare": {"est": 0.1, "se": 0.01, "n_cond": 3}}
    rej, unmeasured = G.compare(ref, est)
    assert rej == ["dir_ok"] and unmeasured == ["rare"]


def test_reversed_actions_swap_opposite_moves_only():
    Q = G.corrupt({"a": np.array([0, 1, 2, 3, 4, 5])}, "reversed_actions", np.random.default_rng(0))
    assert Q["a"].tolist() == [0, 2, 1, 4, 3, 5]


def test_paired_split_keeps_each_physical_episode_on_one_side():
    ep = np.repeat(np.arange(128), 5)
    te = G.episode_split(ep, ep, np.random.default_rng(0))
    both = np.r_[ep, ep]
    assert all(len(set(te[both == e])) == 1 for e in range(128)) and 0 < te.mean() < 1


def test_revisit_age_comes_from_the_compared_clear_sighting():
    # cell (3, 0) is clear at t=0 (terrain 2), covered by an entity at t=1 (terrain 3 underneath), out of view at t=2,
    # back at t=3 showing 3: compared with the t=0 value, so the gap is 3, not 2 (the covered sighting's age)
    pos = np.array([[10, 10], [10, 10], [10, 11], [10, 10]])
    grid = np.zeros((4, 7, 9), np.int8); grid[1, 3, 0] = 1
    view = np.zeros((4, 7, 9), np.int8); view[0, 3, 0] = 2; view[1, 3, 0] = 3; view[3, 3, 0] = 3
    rows = G.revisit_events([{"pos": pos, "grid": grid, "view": view}])
    disagree = rows[rows[:, 2] == 0]
    assert disagree.tolist() == [[0, 3, 0]] and set(rows[rows[:, 2] == 1, 1]) == {2}


def test_diagonal_steps_keep_the_summary_but_trip_the_cardinal_motion_gate():
    # two zombies stepping diagonally in opposite directions keep counts and centroid (summary scope), yet are illegal
    H = 4; pos = np.full((H + 1, 2), 10); body = np.full((H + 1, 4), 9)
    good = np.zeros((H + 1, 7, 9), np.int8); good[:, 2, 3] = 1; good[:, 4, 5] = 1
    bad = good.copy(); bad[1::2] = 0; bad[1::2, 3, 2] = 1; bad[1::2, 3, 6] = 1
    z = np.zeros(H)
    assert np.array_equal(G.summary_vector(good, body, z, z.astype(bool), pos),
                          G.summary_vector(bad, body, z, z.astype(bool), pos))
    pairs = lambda g: G.traj_pairs(g, body, pos, np.zeros(H, int), np.ones(H + 1), 10, 0)
    assert not G.events(pairs(good))["diag_move_zombies"].any()
    assert G.events(pairs(bad))["diag_move_zombies"].all()


def test_traj_pairs_match_build_pairs():
    T, rng = 5, np.random.default_rng(1)
    L = {"ep_T": np.array([T]), "s_pos": np.c_[np.full(T + 1, 20), 20 + np.arange(T + 1)],
         "s_body": np.c_[[9, 9, 7, 7, 7, 5], np.full((T + 1, 3), 9)], "s_light": np.linspace(1, 0, T + 1),
         "t_action": rng.integers(0, 17, T)}
    for m in G.KINDS:
        L[f"s_{m}_p"] = L["s_pos"][:, None, :] + rng.integers(-3, 4, (T + 1, 2, 2))
        L[f"s_{m}_m"] = rng.random((T + 1, 2)) < 0.5
    P = G.build_pairs(L)
    grid = np.concatenate([P["g0"], P["g1"][-1:]])
    Q = G.traj_pairs(grid, L["s_body"], L["s_pos"], L["t_action"], L["s_light"], 10, 0)
    assert all(np.array_equal(P[k], Q[k]) for k in P) and Q["since"].tolist() == [10, 10, 1, 2, 3]
