"""Learn a generated-continuation calibration on unfiltered BC-policy states.

The gate scored terminal BCE only where death varies across actions, which excludes every
all-safe state and raises prevalence from ~0.12 to 0.64-0.92, and it fits its action
marginal on the test labels themselves. A model calibrated on the states a policy actually
visits must lose that comparison. So the calibration is fitted here on ALL sampled states,
on seeds disjoint from every gate and from each other, selected on tune, and read once on
test.

Intercept-only versus affine is chosen on tune, not by preference. If neither serves
all-safe, escape-rich and trap-heavy at once, the honest answer is a separate generated
head, not per-regime constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load, save
from d4mj.config import Config
from d4mj.counterfactual import collect_outcome_forks
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"
EPS = 1e-6
BLOCKS = {"calibrate": range(12_200, 12_232), "tune": range(12_300, 12_316),
          "test": range(12_400, 12_416)}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def bce(z, y):
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))), EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def within_auc(score, truth):
    out = []
    for s, y in zip(score, truth):
        pos, neg = s[y > 0], s[y == 0]
        if len(pos) and len(neg):
            out.append(float((pos[:, None] > neg[None]).mean()
                             + 0.5 * (pos[:, None] == neg[None]).mean()))
    return float(np.mean(out)) if out else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    args = parser.parse_args()
    folder = HERE / f"v2_phase2_{args.arm}"

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    # the calibration buffers postdate this checkpoint, so heads loads non-strictly and
    # the missing keys are asserted to be exactly those two
    payload = torch.load(folder / "phase2_final.pt", weights_only=False)
    world.load_state_dict(payload["modules"]["part0"])
    missing = heads.load_state_dict(payload["modules"]["part1"], strict=False).missing_keys
    assert set(missing) <= {"continuation_scale", "continuation_shift"}, missing
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world, heads, encoder = world.eval(), heads.eval(), encoder.eval()
    gate_config = replace(saved, horizon=saved.direct_rollout)

    data = {}
    for name, seeds in BLOCKS.items():
        seeds = tuple(seeds)
        assert not set(seeds) & set(gate_config.outcome_gate_seeds), "overlaps the live gate"
        forks = collect_outcome_forks(world, encoder, heads,
                                      replace(gate_config, outcome_gate_seeds=seeds))
        death = forks.true_death.float().numpy()
        data[name] = {"x": logit(forks.model_death.numpy()), "y": death,
                      "lethal": death.sum(1)}
        print(f"{args.arm} {name}: {len(seeds)} seeds, {len(death)} states, "
              f"{death.mean():.3f} lethal, {(death.sum(1) == 0).sum()} all-safe", flush=True)

    def fit(kind):
        x, y = data["calibrate"]["x"].ravel(), data["calibrate"]["y"].ravel()
        if kind == "intercept":
            grid = np.linspace(-10, 10, 4001)
            b = float(grid[np.argmin([bce(x + g, y) for g in grid])])
            return 1.0, b
        best, params = np.inf, (1.0, 0.0)
        for a in np.linspace(0.2, 3.0, 57):
            grid = np.linspace(-10, 10, 801)
            losses = [bce(a * x + g, y) for g in grid]
            k = int(np.argmin(losses))
            if losses[k] < best:
                best, params = losses[k], (float(a), float(grid[k]))
        return params

    options = {kind: fit(kind) for kind in ("intercept", "affine")}
    tune = data["tune"]
    scores = {k: bce(s * tune["x"].ravel() + b, tune["y"].ravel())
              for k, (s, b) in options.items()}
    kind = min(scores, key=scores.get)
    scale, shift = options[kind]
    print(f"\n{args.arm}: tune BCE intercept {scores['intercept']:.4f} "
          f"affine {scores['affine']:.4f} -> {kind}, scale {scale:.3f} shift {shift:+.3f}")

    t = data["test"]
    rows = [("all test", np.ones(len(t["y"]), bool)),
            ("  all-safe", t["lethal"] == 0),
            ("  escape-rich", (t["lethal"] >= 1) & (t["lethal"] <= 2)),
            ("  trap-heavy", t["lethal"] >= 14)]
    marginal = data["calibrate"]["y"].mean(0)
    print(f"{'':<20}{'n':>5}{'BCE raw':>9}{'BCE cal':>9}{'pred raw':>10}{'pred cal':>10}"
          f"{'true':>8}{'AUC':>7}{'floor':>7}")
    report = {"arm": args.arm, "kind": kind, "scale": scale, "shift": shift, "strata": {}}
    for name, mask in rows:
        if mask.sum() == 0:
            print(f"  {name:<18}{0:>5}   none"); continue
        x, y = t["x"][mask], t["y"][mask]
        cal = scale * x + shift
        sig = lambda z: 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        floor = np.tile(marginal, (int(mask.sum()), 1))
        print(f"  {name:<18}{int(mask.sum()):>5}{bce(x, y):>9.3f}{bce(cal, y):>9.3f}"
              f"{sig(x).mean():>10.3f}{sig(cal).mean():>10.3f}{y.mean():>8.3f}"
              f"{within_auc(x, y):>7.3f}{within_auc(floor, y):>7.3f}")
        report["strata"][name.strip()] = {
            "n": int(mask.sum()), "bce_raw": bce(x, y), "bce_calibrated": bce(cal, y),
            "predicted_raw": float(sig(x).mean()), "predicted_calibrated": float(sig(cal).mean()),
            "true": float(y.mean()), "auc": within_auc(x, y), "auc_floor": within_auc(floor, y)}

    ok = all(report["strata"][s]["bce_calibrated"] <= report["strata"][s]["bce_raw"] + 0.02
             for s in report["strata"] if s != "all test")
    report["serves_all_regimes"] = bool(ok)
    print(f"\n  one affine serves every regime: {ok}")
    heads.calibrate_generated(scale, shift)
    save(folder / "phase2_calibrated.pt", saved, part0=world, part1=heads)
    (folder / "generated_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"  frozen into {folder.name}/phase2_calibrated.pt")


if __name__ == "__main__":
    main()
