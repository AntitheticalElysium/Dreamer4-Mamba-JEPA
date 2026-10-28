"""Bounded numerical checks for unimplemented M0 contracts; no learning claims."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saccade.eval.calib import reward_bin, REWARD_SUPPORT


def gradient_routes():
    z = torch.tensor([0.8, 1.3], dtype=torch.float64, requires_grad=True)
    noisy = z + torch.tensor([0.2, -0.1], dtype=z.dtype)
    # Prediction 0 uses its own noisy input; prediction 1 also uses past input 0.
    pred = torch.stack([0.7 * noisy[0], 0.7 * noisy[1] + 0.4 * noisy[0]])
    loss = (pred - z.detach()).square()
    label_only = [torch.autograd.grad(v, z, retain_graph=True)[0].tolist() for v in loss]
    zn = noisy.detach()
    w = torch.tensor(0.7, dtype=z.dtype, requires_grad=True)
    pred_det = torch.stack([w * zn[0], w * zn[1] + 0.4 * zn[0]])
    loss_det = (pred_det - z.detach()).square()
    grad_z = torch.autograd.grad(loss_det.sum(), z, allow_unused=True)[0]
    assert abs(label_only[0][0]) > 0
    assert abs(label_only[1][0]) > 0
    assert grad_z is None
    return {"label_only_encoder_jacobian_loss_by_time": label_only,
            "all_noisy_inputs_detached_encoder_gradient": None,
            "interpretation": "Label detach leaves own-frame gradients; noisy-input detach also removes direct past-input gradients."}


def edm():
    s = torch.tensor([.002, .1, 1., 5., 20.], dtype=torch.float64)
    c_skip = 1 / (s.square() + 1)
    c_out = s / (s.square() + 1).sqrt()
    x_noisy = torch.tensor([.3, -.4, .8, 1.1, -.2], dtype=s.dtype)
    x0 = torch.tensor([.2, -.3, .7, .2, .1], dtype=s.dtype)
    raw = torch.tensor([.1, -.2, .3, -.1, .4], dtype=s.dtype)
    denoised = c_skip * x_noisy + c_out * raw
    raw_target = (x0 - c_skip * x_noisy) / c_out
    w = 1 / c_out.square()
    lhs = (raw - raw_target).square()
    rhs = w * (denoised - x0).square()
    err = float((lhs-rhs).abs().max())
    assert err < 1e-10
    return {"sigma": s.tolist(), "clean_x_mse_weights": w.tolist(),
            "raw_F_vs_weighted_x_max_abs_error": err}


def reward():
    # REWARD_SUPPORT is in raw reward units; the existing evaluator symlogs it.
    lo, hi = np.sign(REWARD_SUPPORT) * np.log1p(np.abs(REWARD_SUPPORT))
    idx = int(reward_bin(np.array(0.)))
    edges = np.linspace(lo, hi, 257)
    centre = float((edges[idx] + edges[idx+1]) / 2)
    emitted = math.copysign(math.expm1(abs(centre)), centre)
    assert emitted != 0
    return {"raw_support": REWARD_SUPPORT, "symlog_support": [lo,hi],
            "zero_bin": idx, "zero_bin_symlog_edges": edges[idx:idx+2].tolist(),
            "bin_centre_symlog": centre, "bin_centre_raw_reward": emitted,
            "constant_bias_H16_gamma997": emitted * (1-.997**16)/(1-.997),
            "constant_bias_infinite_survival_gamma997": emitted / (1-.997),
            "limitation": "Conditional on a bin-centre decoder; no M0 sampler has been implemented."}


def patches():
    x = torch.ones(1, 1, 9, 9)
    coverage = F.fold(F.unfold(x, kernel_size=3, padding=1), output_size=(9,9), kernel_size=3, padding=1)[0,0]
    assert coverage[0,0] == 4 and coverage[0,1] == 6 and coverage[1,1] == 9
    return {"coverage": coverage.int().tolist(), "min": int(coverage.min()), "max": int(coverage.max())}


def acuity():
    r = json.load(open("artifacts/eda/system_design_v1/acuity/acuity.json"))["results"]
    out = {}
    kinds = ["zombies", "cows", "skeletons", "arrows"]
    for resolution, seeds in r.items():
        means = {k: sum(x["night_r1_2"]["recall_"+k]["est"] for x in seeds.values()) / len(seeds) for k in kinds}
        ref = r["7"]
        day_ok = True
        for seed,x in seeds.items():
            for key in ["presence_recall", "kind_acc", "terrain_acc"]:
                day_ok &= x["day_r1_4"][key]["est"] >= ref[seed]["day_r1_4"][key]["est"] - .01
            for k in kinds:
                day_ok &= x["day_r1_4"]["recall_"+k]["est"] >= ref[seed]["day_r1_4"]["recall_"+k]["est"] - .02
        out[resolution] = {"day_clause_all_seeds": bool(day_ok), "night_mean_kind_recall": means,
                           "night_clause_at_least095_mean": all(v >= .95 for v in means.values()),
                           "per_kind_precision_saved": any("precision_"+k in x["night_r1_2"] for x in seeds.values() for k in kinds)}
    assert out["3"]["day_clause_all_seeds"]
    assert not any(v["night_clause_at_least095_mean"] for v in out.values())
    return out


def main():
    files = ["saccade/m0.yaml", "saccade/design/M0_CODE.md", "saccade/design/STEP_SPEC.md",
             "saccade/design/CURRICULUM.md", "saccade/design/LOSSES.md", "saccade/design/TEST_SPEC.md",
             "saccade/design/PERCEPTION.md", "saccade/eval/calib.py", "d4mj/lewm.py",
             "artifacts/experiments/20261023_system_design/eda_periphery_acuity.py",
             "artifacts/eda/system_design_v1/acuity/acuity.json"]
    result = {"gradient_routes": gradient_routes(), "edm": edm(), "reward": reward(),
              "patches": patches(), "acuity": acuity(),
              "sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}}
    path = Path(__file__).with_name("contract_results.json")
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("sha256", "acuity", "patches")}, indent=2))
    print("patch coverage:", result["patches"]["min"], "to", result["patches"]["max"])
    print("wrote", path)


if __name__ == "__main__":
    main()
