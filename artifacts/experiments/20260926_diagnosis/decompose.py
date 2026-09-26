"""D17. Does each world predict the action's AVERAGE effect instead of the state-conditional one?

One step, all 17 actions, on the diagnostic futures (D10 fan). True change D = mean over the 4 keys of
s'(a) - s; imagined change G = g(a) - s. Per action a, the action-average m_a = mean over roots of D (and
mG_a of G). Decomposition:
  R2_action_true        share of the true change's variance explained by the action identity alone
  R2_action_imagined    the same for the imagined change
  cond_energy_ratio     sum ||G - mG_a||^2 / sum ||D - m_a||^2: how much state-dependent change the world
                        produces at all, relative to the truth
  captured_marginal     1 - sum_a n_a ||mG_a - m_a||^2 / sum_a n_a ||m_a||^2
  captured_conditional  1 - sum ||(G - mG_a) - (D - m_a)||^2 / sum ||D - m_a||^2
  cond_slope            regression slope of the imagined state-dependent part on the true one
Computed over all actions, and over the non-SLEEP actions (SLEEP's global desaturation dominates energy).
Also, per world: the tail concentration of the one-step error (D11's error_tail_share over tail_var at k=1),
the Mahalanobis radius of true and imagined states, and each world's training loss geometry.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")
LOSS = {"H2": "MSE on raw z (joint), then bridge heads", "Z": "MSE on raw z, var^-1/2 weights",
        "sZ": "L1 on layer-normed z", "U": "MSE on u, var^-1/2 weights (48x range)",
        "W": "MSE on w = u/std (equal)", "T": "L1 on layer-normed tokens"}


def decompose(G, D, acts):
    out = {}
    g, d = G[:, acts], D[:, acts]
    mg, md = g.mean(0, keepdim=True), d.mean(0, keepdim=True)
    total_true = ((d - d.flatten(0, 1).mean(0)) ** 2).sum()
    total_gen = ((g - g.flatten(0, 1).mean(0)) ** 2).sum()
    cond_true, cond_gen = d - md, g - mg
    out["R2_action_true"] = float(1 - (cond_true ** 2).sum() / total_true)
    out["R2_action_imagined"] = float(1 - (cond_gen ** 2).sum() / total_gen)
    out["cond_energy_ratio"] = float((cond_gen ** 2).sum() / (cond_true ** 2).sum())
    out["captured_marginal"] = float(1 - ((mg - md) ** 2).sum() / (md ** 2).sum())
    out["captured_conditional"] = float(1 - ((cond_gen - cond_true) ** 2).sum() / (cond_true ** 2).sum())
    out["cond_slope"] = float((cond_gen * cond_true).sum() / (cond_true ** 2).sum())
    return out


def main():
    compound = json.loads((HERE / "compound.json").read_text())
    result = {}
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        root = d["root"][:, None]
        G = (d["one"] - root).double()
        D = (d["one_true"].mean(1) - root).double()
        row = {"loss": LOSS[w], "all": decompose(G, D, list(range(17))),
               "no_sleep": decompose(G, D, [a for a in range(17) if a != 6])}
        k1 = compound["worlds"][w]["depth"][0]
        row["tail_error_share_k1"] = k1["error_tail_share"]
        row["tail_variance_share"] = k1["tail_var"]
        row["tail_concentration"] = k1["error_tail_share"] / k1["tail_var"]
        row["maha_true_k1"], row["maha_gen_k1"] = k1["maha_true"], k1["maha_gen"]
        row["maha_true_k16"], row["maha_gen_k16"] = compound["worlds"][w]["depth"][15]["maha_true"], compound["worlds"][w]["depth"][15]["maha_gen"]
        result[w] = row
        print(w, json.dumps({k: ({kk: round(vv, 3) for kk, vv in v.items()} if isinstance(v, dict) else
                                 (round(v, 3) if isinstance(v, float) else v)) for k, v in row.items()}), flush=True)
    (HERE / "decompose.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
