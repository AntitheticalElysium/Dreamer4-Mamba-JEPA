"""Fixed-output D versus L1 diagnosis; no model forward, fitting, or GPU.

Score each saved oracle/control trajectory with both objectives. Partition by
physical target layouts and unchanged frozen mob-reader decisions. Correct a
token group to its actual target to measure the objective's exact incentive.
Output gradients are evidence about the loss at fixed outputs, not parameter
updates or the outcome of a retrain. Saved predictions use original fp16 export.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/eda/oracle_v1"
MAPC = [i for i in range(63) if i != 31]
GROUPS = {"1": [0], "2-4": [1, 2, 3], "5-8": [4, 5, 6, 7], "9-16": list(range(8, 16))}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def objectives(e):
    ab = e.abs()
    l1 = ab.flatten(1).mean(-1)
    mse = e.square().flatten(1).mean(-1)
    worst = .01 * e.square().flatten(1).max(-1).values
    return {"l1": l1, "mse": mse, "d_nomax": mse + .1 * l1,
            "max_only": worst, "d": mse + .1 * l1 + worst}


def gradients(e):
    n = e.shape[-1] * e.shape[-2]
    l1 = e.sign() / n
    d = (2 * e + .1 * e.sign()) / n
    ix = e.square().flatten(1).max(-1).indices
    extra = torch.zeros_like(e).flatten(1)
    extra.scatter_(1, ix[:, None], .02 * e.flatten(1).gather(1, ix[:, None]))
    extra = extra.view_as(e)
    return {"l1": l1, "mse": 2 * e / n, "d_nomax": d,
            "max_only": extra, "d": d + extra}, ix // e.shape[-1]


def main():
    torch.set_num_threads(4)
    panel = torch.load(OUT / "panel.pt", map_location="cpu", weights_only=False)
    state = torch.load(OUT / "snap_bank_readers.pt", map_location="cpu", weights_only=False)
    m = state["heads"]["mob"]
    reader_w = (m["w"][:-1] / m["sd"][:, None]).float()
    thresholds = torch.tensor(state["thresholds"])
    def scores(x):
        sh = x.shape[:-1]
        q = x.reshape(-1, 192)
        parts = []
        for i in range(0, len(q), 4096):
            f = ((q[i:i+4096].float() - m["mu"]) / m["sd"]).double()
            parts.append((torch.cat([f, torch.ones(len(f), 1, dtype=torch.float64)], 1) @ m["w"]).float())
        return torch.cat(parts).view(*sh, 4)
    mob_names = {"cow": (1, 2), "zombie": (0, 0)}  # physical-layout, frozen-reader channel
    result = {"schema": "d_loss_fixed_outputs_v1", "unit": "output-gradient, not parameter-gradient",
              "predictions_dtype": "original exported fp16 converted to fp32",
              "inputs": {str(p.relative_to(ROOT)): sha(p) for p in [OUT / "panel.pt", OUT / "snap_bank_readers.pt"]},
              "formula": "D=mean(e^2)+0.1mean|e|+0.01mean_frame(max_element(e^2))", "worlds": {}}
    for world in ("oracle_deltairis", "control_deltairis", "oracle_l1", "control_l1"):
        path = OUT / f"pred_{world}.pt"
        result["inputs"][str(path.relative_to(ROOT))] = sha(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        wr = result["worlds"].setdefault(world, {})
        for mode in ("tf", "gen"):
            pred = saved[mode].float()
            target = panel["tokens"][:, 16:32].float()
            target_scores = scores(target)
            decoded = scores(pred) > thresholds
            e = (pred - target).flatten(0, 1)
            o = objectives(e)
            grad, worst = gradients(e)
            mr = wr.setdefault(mode, {})
            for group, steps in GROUPS.items():
                choose = torch.zeros(200, 16, dtype=torch.bool)
                choose[:, steps] = True
                choose = choose.flatten()
                eg = e[choose]
                og = {k: v[choose] for k, v in o.items()}
                gg = {k: v[choose] for k, v in grad.items()}
                worst_g = worst[choose]
                row = {"frames": int(choose.sum()), "objectives": {k: float(v.mean()) for k, v in og.items()},
                       "max_term_share_of_d": float(og["max_only"].sum() / og["d"].sum()),
                       "max_term_share_of_absolute_d_output_gradient": float(gg["max_only"].abs().sum() / gg["d"].abs().sum()),
                       "worst_coordinate_cell_share": {"hud": float((worst_g >= 63).float().mean()),
                          "player": float((worst_g == 31).float().mean())}, "mobs": {}}
                for mob, (lay_ch, read_ch) in mob_names.items():
                    truth = panel["layouts"][:, 16:32, :, lay_ch] > 0
                    prev = panel["layouts"][:, 15:31, :, lay_ch] > 0
                    truth[:, :, 31] = False
                    prev[:, :, 31] = False
                    de = decoded[:, :, :63, read_ch].clone()
                    de[:, :, 31] = False
                    masks = {"true_present": truth, "false_positive": de & ~truth,
                             "false_negative": truth & ~de, "source_residue": de & prev & ~truth}
                    mb = row["mobs"].setdefault(mob, {})
                    for label, mask in masks.items():
                        mg = torch.zeros(200, 16, 81, dtype=torch.bool)
                        mg[:, :, :63] = mask
                        mg = mg.flatten(0, 1)[choose]
                        count = int(mg.sum())
                        corr = eg.clone()
                        corr[mg] = 0
                        corrected = objectives(corr)
                        sub = {"cells": count,
                               "gradient_abs_share": {k: float((v.abs().sum(-1) * mg).sum() / v.abs().sum()) for k, v in gg.items()},
                               "gradient_l2_energy_share": {k: float((v.square().sum(-1) * mg).sum() / v.square().sum()) for k, v in gg.items()},
                               "exact_loss_gain_from_true_token_substitution": {k: float((og[k] - corrected[k]).mean()) for k in og},
                               "worst_term_owner_frames": int(mg[torch.arange(len(mg)), worst_g].sum())}
                        # A gradient descent step on the emission changes the
                        # linear mob score by -g dot reader_w. Restrict to cells
                        # whose actual target is below the same frozen threshold.
                        valid_fp = mg & (target_scores.flatten(0, 1)[choose, :, read_ch] <= thresholds[read_ch])
                        sub["target_reader_absent_cells"] = int(valid_fp.sum())
                        sub["gradient_step_increases_mob_score_cells"] = {
                            k: int(((-v @ reader_w[:, read_ch]) > 0)[valid_fp].sum()) for k, v in gg.items()}
                        mb[label] = sub
                mr[group] = row
            print(json.dumps({"world": world, "mode": mode, "late": mr["9-16"]}), flush=True)
    # The hand-coded gradients must match autograd, including single max-index
    # ownership rather than amax's tie-splitting rule.
    x = torch.randn(5, 81, 192, generator=torch.Generator().manual_seed(20261012), requires_grad=True)
    gd, _ = gradients(x.detach())
    checks = {}
    for k, loss in objectives(x).items():
        g = torch.autograd.grad(loss.sum(), x, retain_graph=True)[0]
        checks[k] = float((g - gd[k]).abs().max())
        assert checks[k] < 1e-7, checks
    result["autograd_max_difference"] = checks
    result["source_sha256"] = sha(__file__)
    dest = OUT / "d_loss_fixed_outputs.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2)); tmp.replace(dest)
    print(json.dumps({"done": str(dest), "autograd_max_difference": checks}), flush=True)


if __name__ == "__main__":
    main()
