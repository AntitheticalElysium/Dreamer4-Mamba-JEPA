"""Descriptive (EDA, no generation): the coarse-token geometry of the HUD digits in P1-F3 and the training exposure of each
count transition, beside the determinant curves' per-(slot, count) generated success (readings/curves.json).
Per HUD cell and value: the mean real token over the train400 frames and the within-value spread (RMS distance to the
value's mean: lighting / FiLM variation); per adjacent pair of values the separation |mu(n + 1) - mu(n)| / spread.
Exposure: train-split transitions (s0_train_labels) whose count went n -> n + 1 at each slot (any cause), and the
mining increments among them (awake, DO, faced block, tool). Writes readings/geometry.json.
Shared structure (the "+1 per slot, no transfer" reading): per value n, the cosine between the count slots' steps
mu(n + 1) - mu(n), and a linear digit reader ([cell token, v] -> value, logistic) fitted on wood + stone tokens read
on coal / iron tokens (and the reverse), against a reader fitted on the slot itself.
Usage (project venv, GPU): python -m artifacts.experiments.20261106_determinant_curves.geometry
"""
import json
import numpy as np
import torch

from saccade.eval import s1 as S

from .curves import CKPT, HERE, HUD, IF, LABELS, MINING, DO


def shared(p, v, Lt):
    from sklearn.linear_model import LogisticRegression
    slots = {k: MINING[k][2] for k in MINING}
    ep = np.repeat(np.arange(len(Lt["ep_T"])), Lt["ep_T"] + 1)
    fit, test = ep % 10 != 9, ep % 10 == 9
    X = {k: torch.cat([p[:, c], v], 1).numpy() for k, c in slots.items()}
    Y = {k: Lt["s_inv"][:, HUD[c][1]].clip(0, 9) for k, c in slots.items()}
    mu = {k: {n: p[Y[k] == n, c].mean(0) for n in range(10) if (Y[k] == n).sum() >= 20} for k, c in slots.items()}
    step = lambda k, n: mu[k][n + 1] - mu[k][n]
    cos = {f"{a}~{b}": {n: float(torch.nn.functional.cosine_similarity(step(a, n), step(b, n), 0))
                        for n in range(9) if all(m in mu[x] for x in (a, b) for m in (n, n + 1))}
           for a, b in (("wood", "stone"), ("wood", "coal"), ("stone", "coal"), ("coal", "iron"), ("stone", "iron"))}
    read = {}
    for src, dst in ((("wood", "stone"), ("coal", "iron")), (("coal", "iron"), ("wood", "stone")), (("wood",), ("stone",))):
        Xs, Ys = np.concatenate([X[k][fit] for k in src]), np.concatenate([Y[k][fit] for k in src])
        r = LogisticRegression(C=1, max_iter=2000).fit(Xs, Ys)
        read["+".join(src) + "->" + "+".join(dst)] = {k: float((r.predict(X[k][test]) == Y[k][test]).mean()) for k in dst}
    for k in slots:                                                      # same-slot reference
        r = LogisticRegression(C=1, max_iter=2000).fit(X[k][fit], Y[k][fit])
        read[f"{k}->{k}"] = float((r.predict(X[k][test]) == Y[k][test]).mean())
    return {"step_cosine": cos, "digit_reader_transfer": read}


def main():
    model, cfg = IF.load(CKPT)
    Lt = dict(np.load(LABELS / "s0_train400_labels.npz"))
    enc = S.encode(model, Lt, cfg)
    p = torch.cat([e["p"] for e in enc]).float()
    out = {"cells": {}}
    for c, (arr, k) in HUD.items():
        y = torch.as_tensor(Lt[arr][:, k].clip(0, 9))
        mu = {int(n): p[y == n, c].mean(0) for n in y.unique() if (y == n).sum() >= 20}
        spread = {n: float((p[y == n, c] - m).norm(dim=-1).square().mean().sqrt()) for n, m in mu.items()}
        out["cells"][str(c)] = {"frames": {n: int((y == n).sum()) for n in mu}, "spread": spread,
                                "step": {f"{n}->{n + 1}": {"distance": float((mu[n + 1] - mu[n]).norm()),
                                                          "separation": float((mu[n + 1] - mu[n]).norm() / np.sqrt(spread[n] * spread[n + 1]))}
                                         for n in mu if n + 1 in mu},
                                "to_zero": {n: float((mu[n] - mu[0]).norm()) for n in mu if 0 in mu}}
    v = torch.cat([e["v"] for e in enc]).float()
    out["shared"] = shared(p, v, Lt)
    L = dict(np.load(LABELS / "s0_train_labels.npz"))
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"])
    g = s_off[ep] + np.arange(len(ep)) - a_off[ep]
    d = IF.DIRS[L["s_dir"][g]]
    face = L["s_view"][g, 3 + d[:, 0], 4 + d[:, 1]]
    out["exposure"] = {}
    for kind, (block, _, cell, tool) in MINING.items():
        col = HUD[cell][1]
        n0, n1 = L["s_inv"][g, col], L["s_inv"][g + 1, col]
        inc = n1 == n0 + 1
        mine = inc & ~L["s_sleep"][g] & (L["t_action"] == DO) & (face == block)
        out["exposure"][kind] = {int(n): {"increments": int((inc & (n0 == n)).sum()), "mining": int((mine & (n0 == n)).sum())}
                                 for n in range(9)}
    curves = json.loads((HERE / "readings/curves.json").read_text())["curves"]["mining"]
    out["joined"] = [{"kind": k, "n": n, "success": curves[f"{k}_{n}"]["count_next"]["gen"], "histories": curves[f"{k}_{n}"]["histories"],
                      "mining_exposure": out["exposure"][k][n]["mining"],
                      "separation": out["cells"][str(MINING[k][2])]["step"].get(f"{n}->{n + 1}", {}).get("separation")}
                     for k in MINING for n in range(9) if f"{k}_{n}" in curves]
    (HERE / "readings/geometry.json").write_text(json.dumps(out, indent=1))
    for r in out["joined"]:
        print(r, flush=True)
    print(json.dumps(out["shared"]), flush=True)
    for c in ("63", "64", "66"):
        print(c, json.dumps(out["cells"][c]["step"]), flush=True)


if __name__ == "__main__":
    main()
