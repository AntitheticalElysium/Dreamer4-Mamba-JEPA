"""Dissection of M0.5 arm G (user, 10-30: what exactly fails). CPU, saved per-row arrays (`eval/*_guided_*.npz`) and
dev labels only. Per guided-route HUD class and checkpoint, every eligible readable generated glimpse (fixation rebuilt as encode_with draws it: the rule's window, else the
seeded prior) is split into
success (end), copy (start), down (start - 1) and far (any other value); a far value is tested against the REAL values
at t of the other HUD counters inside the same window (the rule's window centre), beside a chance rate from 200
permutations of those neighbours across samples of the class; the outcome split by whether the glimpse was already
on the cell at t - 1; "down" split by whether start - 1 is the counter's own value at t - 2 .. t - P (a stale copy); and the recorded one-step drink changes (labels), overall and on DO-facing-water frames. Rows are rebuilt from labels (evaluate.rows_of) and
checked against the saved start, end and observed values first.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.dissect.error_modes
"""
import importlib, json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M
from saccade.eval import s1 as S

E = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.evaluate")
EV, PM, FC, CAT, P = E.EV, E.PM, E.FC, E.CAT, E.P
HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
LABELS = ("base", "guided_0300", "guided_0750", "guided_1500")
KINDS = ("drink", "wood", "drink_fail", "adjacent", "risky")
COUNTERS = {63: ("s_body", 0), 64: ("s_body", 1), 65: ("s_body", 2), 66: ("s_body", 3), 67: ("s_inv", 5),
            68: ("s_inv", 0), 69: ("s_inv", 1)}          # health, food, drink, energy, sapling, wood, stone (pixel-checked)
PERMS = 200


def main():
    L = dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    dummy = [{"p": range(int(t) + 1)} for t in L["ep_T"]]
    rows, _ = E.rows_of(L, dummy, E.N)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    W = PM.window_of_state(L, rule)
    seed = torch.load(EV.BASE, weights_only=False, map_location="cpu")["cfg"]["p1"]["monitor"]["fixed_fixation_seed"]
    fix = []                                                   # encode_with's fixation of every dev frame (guided route)
    for e, ep in enumerate(S.episodes(L)):
        r = np.random.default_rng([seed, ep.shard, ep.slot])
        fx = [tuple(int(x) for x in D.fixation_prior(r, ())) for _ in range(ep.T + 1)]
        w = W[s_off[e]:s_off[e + 1] - 1]
        fix.append(fx[:1] + [tuple(int(x) for x in w[t - 1]) if w[t - 1, 0] >= 0 else fx[t] for t in range(1, ep.T + 1)])
    rng = np.random.default_rng(0)
    res = {}
    for kind in KINDS:
        info = EV.row_info(kind, rows[kind], L)
        res[kind] = {}
        for label in LABELS:
            z = dict(np.load(EVAL / f"{label}_guided_{kind}.npz"))
            idx = z["row_ctx"]
            assert (z["start"] == [info[i][1] for i in idx]).all() and (z["end"] == [info[i][2] for i in idx]).all()
            assert (z["observed"] == E.observed(kind, rows[kind], idx, L)).all()
            g = np.array([s_off[rows[kind][i][0]] + rows[kind][i][1] for i in idx])      # state index of frame t
            cell = np.array([info[i][0] for i in idx])
            w = np.array([fix[rows[kind][i][0]][rows[kind][i][1]] for i in idx])
            m = z["eligible"] & z["readable"]
            by_rule = W[g - 1, 0] >= 0
            wp = np.array([fix[rows[kind][i][0]][rows[kind][i][1] - 1] for i in idx])[m]
            prev_on = (M.window(torch.as_tensor(wp.astype(np.int64))).numpy() == cell[m, None]).any(1)   # glimpse on it at t - 1
            cells = M.window(torch.as_tensor(w[m].astype(np.int64))).numpy()
            assert (cells == cell[m, None]).any(1).all()
            read, start, end = z["read"][m], z["start"][m], z["end"][m]
            far = (read != start) & (read != end) & (read != start - 1)
            nb = np.full((m.sum(), 9), -9)                                      # real values of window neighbours at t
            for j, c in enumerate(cells.T):
                for k in range(m.sum()):
                    if c[k] in COUNTERS and c[k] != cell[m][k]:
                        arr, col = COUNTERS[int(c[k])]
                        nb[k, j] = int(np.clip(L[arr][g[m][k], col], 0, None))
            hit = (nb == read[:, None]).any(1)
            arr_, col_ = E.EV.HUD_CELLS[E.READS[kind]][1:]
            hist = np.stack([np.clip(L[arr_][g[m] - k, col_], 0, None) for k in range(1, P + 1)], 1)   # value at t - k
            in_hist = lambda x: ((hist[:, 1:] == x[:, None]).any(1)) & (hist[:, 0] != x)          # seen at t-2..t-P, not t-1
            groups = {"appearance": start == 0, "increment": start >= 1} if kind in ("drink", "wood") else {"all": start >= 0}
            res[kind][label] = {}
            for gname, gm in groups.items():
                n = int(gm.sum())
                f = far & gm
                ch = [float((nb[p][f] == read[f][:, None]).any(1).mean()) if f.any() else None
                      for p in (rng.permutation(len(nb)) for _ in range(PERMS))]
                res[kind][label][gname] = {
                    "readable": n, "placed_by_rule": float(by_rule[m][gm].mean()), "success": int((gm & (read == end)).sum()), "copy": int((gm & (read == start)).sum()),
                    "down": int((gm & (read == start - 1) & (read != end)).sum()), "far": int(f.sum()),
                    "far_values": {int(v): int(c) for v, c in zip(*np.unique(read[f], return_counts=True))},
                    "far_equals_window_neighbour": float(hit[f].mean()) if f.any() else None,
                    "chance_mean": float(np.mean(ch)) if f.any() else None,
                    "chance_max": float(np.max(ch)) if f.any() else None,
                    "far_by_start": {int(a): {int(v): int(c) for v, c in zip(*np.unique(read[f & (start == a)], return_counts=True))}
                                     for a in np.unique(start[f])},
                    "down_by_history": {k: {"n": int((gm & hm).sum()), "down": int((gm & hm & (read == start - 1) & (read != end)).sum())}
                                        for k, hm in (("start-1_seen_t-2..t-P", in_hist(start - 1)),
                                                      ("start-1_not_seen", ~in_hist(start - 1)))},
                    "far_seen_in_own_history": float(in_hist(read)[f].mean()) if f.any() else None,
                    "by_previous_glimpse": {k: {"n": int((gm & pm).sum()), "success": int((gm & pm & (read == end)).sum()),
                                                "copy": int((gm & pm & (read == start)).sum()),
                                                "down": int((gm & pm & (read == start - 1) & (read != end)).sum())}
                                            for k, pm in (("on_cell", prev_on), ("elsewhere", ~prev_on))}}
    # recorded rates (labels): the drink counter's one-step changes, overall and on DO-facing-water frames
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    a_off = np.r_[0, np.cumsum(L["ep_T"])]
    g0 = np.concatenate([np.arange(s_off[e], s_off[e + 1] - 1) for e in range(len(L["ep_T"]))])
    act = np.concatenate([L["t_action"][a_off[e]:a_off[e + 1]] for e in range(len(L["ep_T"]))])
    dk = np.clip(L["s_body"][g0 + 1, 2], 0, None) - np.clip(L["s_body"][g0, 2], 0, None)
    v0 = np.clip(L["s_body"][g0, 2], 0, None)
    dw = (act == CAT.DO) & (face[g0] == CAT.WATER) & (L["s_sleep"][g0] == 0)
    res["recorded_drink_changes"] = {k: {"frames": int(mk.sum()), "down": float((dk[mk] < 0).mean()), "up": float((dk[mk] > 0).mean()),
                                         "same": float((dk[mk] == 0).mean())}
                                     for k, mk in (("all_v_ge_1", v0 >= 1), ("do_water_v_1_8", dw & (v0 >= 1) & (v0 <= 8)),
                                                   ("do_water_v_0", dw & (v0 == 0)), ("do_water_v_9", dw & (v0 == 9)))}
    prev_dw = np.r_[False, dw[:-1]] & np.r_[False, np.diff(g0) == 1]
    res["recorded_drink_changes"]["do_water_v_1_8_previous_frame_also_do_water"] = float(prev_dw[dw & (v0 >= 1) & (v0 <= 8)].mean())
    res["recorded_drink_changes"]["do_water_v_0_previous_frame_also_do_water"] = float(prev_dw[dw & (v0 == 0)].mean())
    (HERE / "error_modes.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["recorded_drink_changes"], indent=1))
    for kind, r in res.items():
        if kind == "recorded_drink_changes":
            continue
        for label, gs in r.items():
            for gname, x in gs.items():
                print(f"{kind:10s} {gname:10s} {label:12s} n {x['readable']:4d} succ {x['success']:4d} copy {x['copy']:4d} "
                      f"down {x['down']:4d} far {x['far']:4d} far=neighbour {x['far_equals_window_neighbour']} "
                      f"chance {x['chance_mean']} (max {x['chance_max']})  far values {x['far_values']}")
                print("      by previous glimpse", x["by_previous_glimpse"], " far by start", x["far_by_start"])
                print("      down by history", x["down_by_history"], " far seen in own history", x["far_seen_in_own_history"])


if __name__ == "__main__":
    main()
