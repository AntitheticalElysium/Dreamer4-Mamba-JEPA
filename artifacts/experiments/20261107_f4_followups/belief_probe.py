"""Belief probe (PLAN.md 9): is each hidden timer's due time, and zombie readiness, linearly present in bp_t (the
denoiser's memory input for frame t)? Either code tree (F4 canonical, F3 under fa6edd1a's core). Frozen: S1's encode,
exp 3's logistic probe with 5 episode folds, out-of-fold AP with episode bootstrap.
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.belief_probe CKPT OUT"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M
from saccade.eval import s1 as S

DC = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_checks")
DR = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_readings")
CV, RT = DC.CV, DC.RT
CAP = 40000


def main(ckpt, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(ckpt)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    j_all = np.arange(len(T["g"]))
    bp = lambda j: torch.stack([enc[int(T["ep"][x])]["bp"][int(T["t"][x])] for x in j]).double()
    if True:
        @torch.no_grad()
        def percept(j):                                                  # frame t - 1's embedding + the action: no history
            P = M.Percept(*(torch.stack([enc[int(T["ep"][x])][k][int(T["t"][x]) - 1] for x in j]).float().to(CV.DEV) for k in ("p", "g", "v")),
                          torch.stack([enc[int(T["ep"][x])]["fix"][int(T["t"][x]) - 1] for x in j]).to(CV.DEV))
            emb = torch.cat([model.belief.percept_embed(M.Percept(*(z[a:a + 4096] for z in P.parts()), P.fix[a:a + 4096])).float().cpu()
                             for a in range(0, len(j), 4096)])
            return torch.cat([emb, torch.nn.functional.one_hot(torch.as_tensor(T["action"][j]), 17).float()], 1).double().detach()
    q, body, rng = DC.quiet(L, T), T["body"], np.random.default_rng(2114)
    met = (body[:, 1] > 0) & (body[:, 2] > 0) & (body[:, 3] > 0)
    need = (body[:, 1] == 0) | (body[:, 2] == 0) | (body[:, 3] == 0)
    tb = T["true_body"]
    timers = {"food_down_awake": (q & ~T["asleep"] & (body[:, 1] > 0), 1, -1), "drink_down_awake": (q & ~T["asleep"] & (body[:, 2] > 0), 2, -1),
              "energy_down_awake": (q & ~T["asleep"] & (body[:, 3] > 0), 3, -1), "energy_up_asleep": (q & T["asleep"] & (body[:, 3] < 9), 3, 1),
              "recovery_awake": (q & ~T["asleep"] & met & (body[:, 0] < 9), 0, 1), "need_damage_awake": (q & ~T["asleep"] & need, 0, -1)}
    res = {"checkpoint": ckpt, "tree": "fa6edd1a" if DC.OLD else "canonical", "targets": {}}

    def probe(X, y, ep):
        folds, s = RT.folds_of(ep), np.zeros(len(y))
        for f in range(5):
            s[folds == f] = RT.logistic(X[folds != f], y[folds != f], X[folds == f])
        return s

    for name, (m, col, d) in timers.items():
        j = np.nonzero(m)[0]
        j = np.sort(rng.choice(j, CAP, replace=False)) if len(j) > CAP else j
        y = (tb[j, col] == body[j, col] + d).astype(float); ep = T["ep"][j]
        since = DR.since_last(L, T, col, d)[j]
        bins = np.array([next((i for i, (a, b) in enumerate(DR.BINS) if a <= x <= b), len(DR.BINS)) if x >= 0 else len(DR.BINS) for x in since])
        ref = torch.nn.functional.one_hot(torch.as_tensor(bins), len(DR.BINS) + 1).double()
        scores = {"belief_bp": probe(bp(j), y, ep), "percept_and_action": probe(percept(j), y, ep), "visible_history_since_bins": probe(ref, y, ep)}
        names = [f"{a}-{b}" if b < 10 ** 6 else f"{a}+" for a, b in DR.BINS] + ["none"]
        res["targets"][name] = {"rows": int(len(j)), "episodes": int(len(np.unique(ep))), "prevalence": float(y.mean()),
                                "ap": {k: RT.ap_boot(y, s, ep) for k, s in scores.items()},
                                "profile (bin: true, belief, percept)": {names[b]: [float(y[bins == b].mean()), float(scores["belief_bp"][bins == b].mean()),
                                                                                    float(scores["percept_and_action"][bins == b].mean())]
                                                                         for b in range(len(names)) if (bins == b).sum() >= 50}}
        print(name, {k: round(v["ap"], 3) for k, v in res["targets"][name]["ap"].items()}, "prev", round(y.mean(), 4), flush=True)
    adj = T["adjacent"] & (T["t"] >= CV.P)
    for name, m, ytrue, heur in (("zombie_ready_awake", adj & ~T["asleep"], T["ready"], "ready"),
                                 ("zombie_hit_awake", adj & ~T["asleep"], T["true_body"][:, 0] < body[:, 0], "ready"),
                                 ("zombie_hit_asleep", adj & T["asleep"], T["true_body"][:, 0] < body[:, 0], "ready")):
        j = np.nonzero(m)[0]
        j = np.sort(rng.choice(j, CAP, replace=False)) if len(j) > CAP else j
        y, ep = ytrue[j].astype(float), T["ep"][j]
        scores = {"belief_bp": probe(bp(j), y, ep), "percept_and_action": probe(percept(j), y, ep),
                  "visible_history_heuristic": RT.observable_heuristic(L, T, j, heur)}
        res["targets"][name] = {"rows": int(len(j)), "episodes": int(len(np.unique(ep))), "prevalence": float(y.mean()),
                                "ap": {k: RT.ap_boot(y, s, ep) for k, s in scores.items()}}
        print(name, {k: round(v["ap"], 3) for k, v in res["targets"][name]["ap"].items()}, "prev", round(y.mean(), 4), flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
