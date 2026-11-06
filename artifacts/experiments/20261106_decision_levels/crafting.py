"""Crafting at the frame level (PLAN.md, last section): does frozen P1-F3 draw a crafted tool, and only when the craft
is feasible? Feasibility lives in OTHER tokens: a table (iron tools: and a furnace) among the 8 cells around the player,
the materials in their HUD slots (Craftax Classic game_logic.do_crafting; the rule is asserted against the simulator's
outcome on every row). One-step futures with T2's prefix-20 start and readers, plus readers for the six tool slots.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_decision_levels.crafting
  [--smoke] [--summarise]
"""
import importlib, json, os, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

F_ = importlib.import_module("artifacts.experiments.20261106_decision_levels.frozen")
CV = F_.CV
HERE = Path(__file__).resolve().parent
TOOLS = {11: (6, 73, 5), 12: (7, 74, 13), 13: (8, 75, 20), 14: (9, 76, 6), 15: (10, 77, 14), 16: (11, 78, 21)}  # action: s_inv column, HUD cell, achievement
COST = {11: (0,), 12: (0, 1), 13: (0, 1, 2, 3), 14: (0,), 15: (0, 1), 16: (0, 1, 2, 3)}   # s_inv columns, 1 each
MAT = {0: 68, 1: 69, 2: 70, 3: 71}                                       # material column -> HUD cell
TABLE, FURNACE, IRON = 11, 12, (13, 16)
CAPS = {"made0": None, "madeN": None, "nostation": 250, "nomaterial": 250, "neither": 150}
INFEASIBLE = ("nostation", "nomaterial", "neither")
HUD = {**CV.HUD, **{cell: ("s_inv", col) for col, cell, _ in TOOLS.values()}}


def strata(T, L):
    """Per transition: '<action>_<stratum>' for awake crafting attempts at t >= P, else ''."""
    nb = np.delete(L["s_view"][T["g"], 2:5, 3:6].reshape(len(T["g"]), 9), 4, 1)
    table, furnace = (nb == TABLE).any(1), (nb == FURNACE).any(1)
    lab = np.full(len(T["g"]), "", object)
    for a, (col, _, _) in TOOLS.items():
        m = (T["t"] >= CV.P) & ~T["asleep"] & (T["action"] == a)
        mat, st = (T["inv"][:, list(COST[a])] >= 1).all(1), table & furnace if a in IRON else table
        assert not (m & ((mat & st) != (T["true_inv"][:, col] > T["inv"][:, col]))).any(), "rule differs from the simulator"
        z = T["inv"][:, col] == 0
        for name, s in (("made0", mat & st & z), ("madeN", mat & st & ~z), ("nostation", mat & ~st & z),
                        ("nomaterial", ~mat & st & z), ("neither", ~mat & ~st & z)):
            lab[m & s] = f"{a}_{name}"
    return lab


def main(smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out = HERE / "readings" / ("crafting_smoke" if smoke else "crafting"); out.mkdir(parents=True, exist_ok=True)
    model, cfg, L = F_.setup()
    L = S.first_episodes(L, 60 if smoke else 0)
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    lab = strata(T, L)
    rng, keep = np.random.default_rng(2108), []
    for s in sorted(set(lab) - {""}):
        idx, cap = np.nonzero(lab == s)[0], CAPS[s.split("_")[1]]
        keep.append(idx if cap is None or len(idx) <= cap else rng.choice(idx, cap, replace=False))
    rows = np.sort(np.concatenate(keep))
    CV.HUD = HUD                                                         # fit_readers / read look HUD up at call time
    with torch.enable_grad():
        R_, valid = CV.fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    s1 = S.load_reader(CV.S1_READER, TR.ROOT / CV.CKPT, CV.DEV)
    orig, M.sample = M.sample, F_.prefix_sample
    try:
        o = CV.generate(model, cfg, enc, L, T, rows, 4200, R_, s1)
    finally:
        M.sample = orig
    o["stratum"] = lab[o["j"]].astype(str)
    np.savez_compressed(out / "raw.npz", **o)
    res = {"checkpoint": CV.CKPT, "recipe": "prefix-20 (T2)", "K": CV.K, "readers": valid,
           "populations": {s: int((lab[rows] == s).sum()) for s in sorted(set(lab[rows]))}}
    (out / "run.json").write_text(json.dumps(res, indent=1, default=str))
    summarise(out, smoke)


def summarise(out, smoke):
    o, res = dict(np.load(out / "raw.npz")), json.loads((out / "run.json").read_text())
    T = CV.transitions(S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 60 if smoke else 0))
    j, ep = o["j"], T["ep"][o["j"]]
    act = np.array([int(s.split("_")[0]) for s in o["stratum"]]); kind = np.array([s.split("_")[1] for s in o["stratum"]])
    ix, r = {c: i for i, c in enumerate(HUD)}, np.arange(len(j))
    val = lambda w, cells: np.stack([o[f"{w}_hud{c}"] for c in HUD])[[ix[c] for c in cells], r]
    col = np.array([TOOLS[a][0] for a in act]); n, nt = T["inv"][j, col], T["true_inv"][j, col]
    tool = {w: val(w, [TOOLS[a][1] for a in act]) for w in ("gen", "rec")}
    spent = {w: np.all([np.where([m in COST[a] for a in act], val(w, [MAT[m]] * len(j)) == T["inv"][j, m] - 1, True)
                        for m in MAT], 0) for w in ("gen", "rec")}
    true_spent = np.all([np.where([m in COST[a] for a in act], T["true_inv"][j, m] == T["inv"][j, m] - 1, True) for m in MAT], 0)
    ev = {"tool_next": (tool["gen"] == n + 1, tool["rec"] == n + 1, nt == n + 1), "tool_copy": (tool["gen"] == n, tool["rec"] == n, nt == n),
          "spent": (spent["gen"], spent["rec"], true_spent),
          "coherent": ((tool["gen"] == n + 1) & spent["gen"], (tool["rec"] == n + 1) & spent["rec"], (nt == n + 1) & true_spent)}
    first = ~T["ach"][j, [TOOLS[a][2] for a in act]]
    rates, rd = {}, {}
    for a in TOOLS:
        rates[a] = {}
        for s in CAPS:
            m = (act == a) & (kind == s)
            if m.any():
                rates[a][s] = {"futures": int(m.sum()), **{k: {w: G.estimate(np.asarray(x, float), m, ep)["est"] for w, x in zip(("gen", "rec_reader", "true"), v)}
                                                          for k, v in ev.items()}}
        m1, m0 = (act == a) & (kind == "made0"), (act == a) & np.isin(kind, INFEASIBLE)
        rd[a] = {"appearance": CV.contrast(ev["tool_next"][0], ev["tool_next"][2], ep, m1, m0),
                 "spent": CV.contrast(ev["spent"][0], ev["spent"][2], ep, m1, m0),
                 "head_unlock_gen": CV.contrast(o["gen_unlock_p"], o["rec_event"].astype(float), ep, m1 & first, m0 & first),
                 "head_unlock_rec": CV.contrast(o["rec_head_unlock_p"], o["rec_event"].astype(float), ep, m1 & first, m0 & first),
                 "world_vs_head": CV.agreement(ev["tool_next"][0], o["gen_unlock_p"], (act == a) & first)}
        rd[a]["prediction"] = ("REFUTES the law's exposure threshold" if rd[a]["appearance"]["verdict"] == "TRACKS" else
                               "as the law predicts" if rd[a]["appearance"]["Q"]["est"] <= 0.3 else "not TRACKS, but Q > 0.3") if a in IRON else "-"
    (out / "crafting.json").write_text(json.dumps({**res, "readings": rd, "rates": rates}, indent=1, default=str))
    for a in TOOLS:
        x = rd[a]["appearance"]
        print(a, x["verdict"], {k: round(x[k]["est"], 3) for k in ("T", "Q", "false")},
              {s: round(rates[a][s]["tool_next"]["gen"], 3) for s in rates[a]}, "spent", rd[a]["spent"]["verdict"],
              "head", rd[a]["world_vs_head"]["verdict"], rd[a]["prediction"], flush=True)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    if "--summarise" in sys.argv:
        summarise(HERE / "readings" / ("crafting_smoke" if smoke else "crafting"), smoke)
    else:
        main(smoke)
