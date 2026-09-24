"""Phase 1: is the representation readable for the within-root decision?

`death` is exactly `successor health <= 0` on this panel -- verified, 100% agreement over every
(root, action) pair. Health is drawn in the Craftax HUD, so the question "can this rung choose
the safe action" decomposes into something far more specific: does the rung still carry the
successor's health, and can a decoder trained for the *decision* recover it?

So each rung is scored three ways on identical rows:

  choice   safe-action choice and within-root AUC under bce6 / bce_death / rank supervision
  health   direct regression of successor health, the quantity `death` is defined from
  control  the matched baselines a world model must beat to have added anything

Stages publish independently and resume, so a later stage never silently reuses a stale earlier
one. Nothing here trains an encoder or a world model.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.m03.cache import resolve_payload
from readout import (DEATH, Fit, evaluate, fit_head, one_hot_actions, opportunity_counts,
                     scores_of, standardize, summarize)

SIDECAR = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/sidecar/sidecar.probe_only.pt"
DIRECT = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2/features/direct_mamba"
OUTCOMES = ("death", "damage", "reward_positive", "achievement_event", "inventory_changed", "tile_changed")
SEEDS = (0, 1, 2)
FAMILIES = ("logistic", "mlp128", "mlp512x2")
PRIMARY_ARMS = ("mamba_raw", "mamba_tc", "direct")


def load_taps(name):
    return torch.load(HERE / f"cache/taps.{name}.pt", map_location="cpu", weights_only=False)


def direct_features(split):
    payload = resolve_payload(torch.load(f"{DIRECT}.{split}.pt", map_location="cpu", weights_only=False))
    return payload["features"]


def successor_rung(arm, taps, split, tap="z"):
    """[N, 17, D] successor features for an arm. Direct's come from its native encoding."""
    if arm == "direct":
        return direct_features(split)["observed_successor"].float()
    value = taps[arm][split]["successor"][tap].float()
    return value.flatten(2) if value.dim() == 4 else value


def root_rung(arm, taps, split, tap="z"):
    if arm == "direct":
        return direct_features(split)["projected"].float()
    value = taps[arm][split]["root"][tap].float()
    return value.flatten(1) if value.dim() == 3 else value


def with_action(x):
    return torch.cat((x, one_hot_actions(len(x))), -1)


def inner_folds(episodes, folds=4, seed=20260919):
    """Grouped inner-TRAIN folds: an episode never straddles the fit/val boundary."""
    keys = np.unique(episodes.numpy())
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    chunks = np.array_split(keys, folds)
    return [(torch.tensor(~np.isin(episodes.numpy(), chunk)),
             torch.tensor(np.isin(episodes.numpy(), chunk))) for chunk in chunks]


def run_fit(xtr, ytr, xdv, ydv, *, family, objective, seed, spec, device, permute=False, inner=None):
    model, curve, params = fit_head(xtr, ytr, family=family, objective=objective, seed=seed,
                                    spec=spec, device=device, inner=inner, permute_labels=permute)
    train_metrics = evaluate(model, xtr, ytr, device)
    dev_metrics = evaluate(model, xdv, ydv, device)
    return {"family": family, "objective": objective, "seed": seed, "parameters": params,
            "train": train_metrics, "dev": dev_metrics, "curve": curve}


def health_regression(xtr, htr, xdv, hdv, *, family, seed, spec, device):
    """R^2 of successor health -- the quantity `death` is literally thresholded from."""
    from readout import build_head
    generator = torch.Generator(device="cpu").manual_seed(seed)
    model = build_head(family, xtr.shape[2:], 1, generator, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    mean, scale = htr.mean(), htr.std().clamp_min(1e-6)
    for step in range(spec.steps):
        index = torch.randint(len(xtr), (min(spec.batch, len(xtr)),), generator=generator)
        x, y = xtr[index].to(device), ((htr[index] - mean) / scale).to(device)
        loss = torch.nn.functional.mse_loss(model(x)[..., 0], y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        pred = torch.cat([model(xdv[i:i + 128].to(device))[..., 0].cpu() for i in range(0, len(xdv), 128)])
    pred = pred * scale + mean
    residual = ((pred - hdv) ** 2).mean()
    total = ((hdv - hdv.mean()) ** 2).mean()
    return {"r2": float(1 - residual / total), "rmse": float(residual.sqrt()),
            "parameters": sum(p.numel() for p in model.parameters())}


def stage(path, compute):
    if path.exists():
        return json.loads(path.read_text())
    result = compute()
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args(argv)
    (args.out / "phase1").mkdir(parents=True, exist_ok=True)
    out = args.out / "phase1"
    device = args.device

    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    taps = {name: load_taps(name) for name in ("mamba_raw", "mamba_tc", "transformer_raw", "transformer_tc")}
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    health = {s: side[s]["next_continuous"][:, :, 0].float() for s in ("train", "dev")}
    spec = Fit(steps=args.steps)

    # ---- label sanity: the exact simulator predicate, not a learned score -------------------
    def label_sanity():
        agree = float(((health["dev"] <= 0) == y["dev"][..., DEATH].bool()).float().mean())
        # `summarize` takes an argmin, so the oracle must score LOW where risk is low:
        # -health ranks the healthiest successor first. Passing health directly selects the
        # MOST fatal fork and scores 0/36 -- a perfect anti-oracle, not a positive control.
        exact = summarize(-health["dev"], y["dev"])
        anti = summarize(health["dev"], y["dev"])
        oracle = summarize(-(~y["dev"][..., DEATH].bool()).float(), y["dev"])
        return {"death_equals_health_le_0": agree,
                "choice_by_true_health": exact,
                "choice_by_true_health_inverted_anti_oracle": anti,
                "choice_by_true_label": oracle,
                "note": "replay/label sanity, NOT a learned-representation score"}
    sanity = stage(out / "label_sanity.json", label_sanity)
    print(json.dumps({"stage": "label_sanity", "agreement": sanity["death_equals_health_le_0"],
                      "oracle_choice": sanity["choice_by_true_label"]["safe_choice"]}), flush=True)

    # ---- anchor: reproduce the published 200-step protocol ---------------------------------
    def anchors():
        rows = []
        for arm in PRIMARY_ARMS:
            xtr, xdv = (with_action(successor_rung(arm, taps, s)) for s in ("train", "dev"))
            xtr, xdv = standardize(xtr, xdv)
            for family in ("logistic", "mlp128"):
                rows.append({"arm": arm, **run_fit(xtr, y["train"], xdv, y["dev"], family=family,
                                                   objective="bce6", seed=0,
                                                   spec=Fit(steps=200), device=device)})
        return {"protocol": "published: 200 steps, six-target BCE, successor+action", "rows": rows}
    anchor = stage(out / "anchors.json", anchors)
    for row in anchor["rows"]:
        print(json.dumps({"stage": "anchor", "arm": row["arm"], "family": row["family"],
                          "dev_safe": f"{row['dev']['safe_choice']}/{row['dev']['opportunity_roots']}",
                          "auc": round(row["dev"]["within_root_auc"], 3)}), flush=True)

    # ---- timing benchmark, to set the budget from measurement ------------------------------
    def timing():
        xtr, xdv = (with_action(successor_rung("mamba_raw", taps, s)) for s in ("train", "dev"))
        xtr, xdv = standardize(xtr, xdv)
        marks = {}
        for family in FAMILIES:
            start = time.time()
            run_fit(xtr, y["train"], xdv, y["dev"], family=family, objective="rank", seed=0,
                    spec=spec, device=device)
            marks[family] = round(time.time() - start, 2)
        return {"seconds_per_fit_at_steps": args.steps, "by_family": marks}
    marks = stage(out / "timing.json", timing)
    print(json.dumps({"stage": "timing", **marks}), flush=True)

    # ---- the 54-fit ladder: arms x families x objectives x seeds, successor-only ------------
    def ladder():
        rows = []
        for arm in PRIMARY_ARMS:
            base = {s: successor_rung(arm, taps, s) for s in ("train", "dev")}
            for action_input in (False, True):
                xtr, xdv = (with_action(base[s]) if action_input else base[s] for s in ("train", "dev"))
                xtr, xdv = standardize(xtr, xdv)
                for family in FAMILIES:
                    for objective in ("bce_death", "rank"):
                        for seed in SEEDS:
                            rows.append({"arm": arm, "action_input": action_input,
                                         **run_fit(xtr, y["train"], xdv, y["dev"], family=family,
                                                   objective=objective, seed=seed, spec=spec,
                                                   device=device)})
                            r = rows[-1]
                            print(json.dumps({"stage": "ladder", "arm": arm, "action": action_input,
                                              "family": family, "objective": objective, "seed": seed,
                                              "dev_safe": f"{r['dev']['safe_choice']}/{r['dev']['opportunity_roots']}",
                                              "auc": round(r["dev"]["within_root_auc"], 3)}), flush=True)
        return {"rows": rows}
    stage(out / "ladder.json", ladder)

    # ---- health decodability: the quantity death is defined from ---------------------------
    def health_stage():
        rows = []
        for arm in PRIMARY_ARMS:
            for tap in (("z",) if arm == "direct" else ("z", "cls", "pooled4", "patches")):
                xtr, xdv = (successor_rung(arm, taps, s, tap) for s in ("train", "dev"))
                xtr, xdv = standardize(xtr, xdv)
                for family in ("logistic", "mlp128"):
                    r = health_regression(xtr, health["train"], xdv, health["dev"],
                                          family=family, seed=0, spec=spec, device=device)
                    rows.append({"arm": arm, "tap": tap, "family": family, "dim": int(xtr.shape[-1]), **r})
                    print(json.dumps({"stage": "health", **rows[-1]}), flush=True)
        return {"rows": rows}
    stage(out / "health.json", health_stage)

    print(json.dumps({"status": "phase1_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
