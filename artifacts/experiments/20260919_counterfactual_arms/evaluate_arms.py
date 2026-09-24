"""Stage 3: score the three arms on the panels the localization ladder established.

Predeclared, before any arm is scored:

  PRIMARY   generated-state safe choice on the 100-root historical confirmation panel, and its
            paired episode-cluster interval against that arm's OWN root+action control.
            Meaningful improvement was declared at +10 points in the plan.
  SECOND    the same on the 36-root DEV panel, for continuity with the ladder.
  FIDELITY  frozen-head action derangement -- a world that has learned action effects must lose
            AUC when its branches are re-labelled.
  REGRESSION factual next-latent MSE on held-out windows, so a win cannot come from abandoning
            the factual objective.

The readout is the one phase 1 selected: within-root pair ranking, mlp128, three seeds. The
root+action control uses the FROZEN encoder's root latents, so it is identical across arms by
construction -- every arm is measured against the same bar.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
LADDER = ROOT / "artifacts/experiments/20260919_localization_ladder"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LADDER))

from d4mj.data import _sha256
from d4mj.m03.cache import resolve_payload
from d4mj.m03.gate import M03Settings, load_m03_bundle
from readout import Fit, fit_head, one_hot_actions, scores_of, standardize, summarize

RAW = ROOT / "artifacts/lewm_gates_20260906/paired/raw/joint/step-010000.pt"
DATASET = ROOT / "artifacts/craftax_support_v2/manifest.json"
EVAL = ROOT / "artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2"
SIDECAR = EVAL / "sidecar/sidecar.probe_only.pt"
SEEDS = (0, 1, 2)


@torch.inference_mode()
def generated(bundle, split, settings, batch=16):
    """The gate's own branch fan: prefill the context, repeat, advance one step per action."""
    context, actions = split["context"], split["past_actions"]
    device = bundle.device
    all_actions = torch.arange(bundle.n_actions, device=device, dtype=torch.long)
    gen, root = [], []
    for start in range(0, len(context), batch):
        end = min(len(context), start + batch)
        n = end - start
        frame = context[start:end, -settings.lewm_context:].to(device)
        past = actions[start:end, -settings.lewm_context + 1:].to(device)
        z, _ = bundle.encoder.projected_and_cls(frame)
        state = bundle.prefill(z, past)
        branches = bundle.repeat_state(state, bundle.n_actions)
        action = all_actions.repeat(n)[:, None]
        predicted, _ = bundle.advance(branches, action)
        gen.append(predicted.latent[:, 0].reshape(n, bundle.n_actions, -1).cpu())
        root.append(z[:, -1, 0].cpu())
    return torch.cat(gen), torch.cat(root)


def confirmation_index(minimum=100):
    """The ladder's declared rule, reused verbatim: ascending seed order, roots with both."""
    shards = sorted((EVAL / "historical/features").glob("replay.*.pt"),
                    key=lambda p: int(p.stem.split(".")[1]))
    kept = []
    for path in shards:
        payload = resolve_payload(torch.load(path, map_location="cpu", weights_only=False))["features"]
        labels = payload["outcomes"][..., 0].bool()
        usable = labels.any(1) & (~labels).any(1)
        if not bool(usable.any()):
            continue
        rows = torch.where(usable)[0]
        kept.append({"seed": int(path.stem.split(".")[1]), "rows": rows.tolist(),
                     "context": payload["context"][rows].clone(),
                     "past_actions": payload["past_actions"][rows].clone(),
                     "outcomes": payload["outcomes"][rows].clone()})
        if sum(len(k["rows"]) for k in kept) >= minimum:
            break
    return kept


def score(xtr, ytr, xdv, ydv, spec, device, order):
    xtr_s, xdv_s = standardize(xtr, xdv)
    intact, frozen = [], []
    for seed in SEEDS:
        model, _, _ = fit_head(xtr_s, ytr, family="mlp128", objective="rank", seed=seed,
                               spec=spec, device=device)
        s = scores_of(model, xdv_s, device)
        intact.append(summarize(s, ydv))
        frozen.append(summarize(scores_of(model, xdv_s[:, order], device), ydv))
    return {"safe": [m["safe_choice"] for m in intact],
            "mean_safe": round(float(np.mean([m["safe_choice"] for m in intact])), 2),
            "opportunity_roots": intact[0]["opportunity_roots"],
            "mean_auc": round(float(np.mean([m["within_root_auc"] for m in intact])), 4),
            "frozen_permuted_mean_auc": round(float(np.mean([m["within_root_auc"] for m in frozen])), 4),
            "frozen_auc_cost": round(float(np.mean([m["within_root_auc"] for m in intact]))
                                     - float(np.mean([m["within_root_auc"] for m in frozen])), 4)}


def paired(a_scores, b_scores, y, episodes, draws=1000, seed=20260919):
    labels = y[..., 0].bool()
    usable = labels.any(1) & (~labels).any(1)

    def rate(s, rows):
        r = rows[usable[rows]]
        return None if len(r) == 0 else float((~labels)[r, s[r].argmin(1)].float().mean())

    groups = [torch.where(episodes == k)[0] for k in episodes.unique(sorted=True)]
    rng = torch.Generator().manual_seed(seed)
    base = rate(a_scores, torch.arange(len(episodes))) - rate(b_scores, torch.arange(len(episodes)))
    samples = []
    for _ in range(draws):
        rows = torch.cat([groups[i] for i in torch.randint(len(groups), (len(groups),), generator=rng)])
        x, z = rate(a_scores, rows), rate(b_scores, rows)
        if x is not None and z is not None:
            samples.append(x - z)
    lo, hi = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"difference": round(base, 4), "interval": [round(lo, 4), round(hi, 4)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--step", type=int, default=10000)
    parser.add_argument("--steps-probe", type=int, default=2000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    settings, spec = M03Settings(), Fit(steps=args.steps_probe)
    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    gen_order = torch.Generator().manual_seed(515)
    from phase4 import derangement
    order = derangement(17, gen_order)

    index = confirmation_index()
    conf_truth = torch.cat([k["outcomes"] for k in index]).float()
    conf_episodes = torch.tensor([k["seed"] for k in index for _ in k["rows"]])
    conf_split = {"context": torch.cat([k["context"] for k in index]),
                  "past_actions": torch.cat([k["past_actions"] for k in index])}
    print(json.dumps({"stage": "panel", "confirmation_roots": len(conf_truth),
                      "seeds": len(index)}), flush=True)

    factual = torch.load(args.cache / "factual.pt", map_location="cpu", weights_only=False)
    held = slice(len(factual["z"]) - 4096, len(factual["z"]))     # never sampled during training
    report = {"schema": "d4mj_cf_arms_eval_v1", "step": args.step,
              "predeclared": {"primary": "generated safe choice on the 100-root confirmation panel, "
                                         "paired vs the arm's OWN root+action control",
                              "meaningful": "+10 points", "readout": "rank / mlp128 / 3 seeds"},
              "arms": {}}
    controls = {}
    for arm in sorted(p.name for p in args.runs.iterdir() if p.is_dir()):
        path = args.runs / arm / f"world_{args.step:06d}.pt"
        if not path.exists():
            print(json.dumps({"stage": "skip", "arm": arm}), flush=True)
            continue
        bundle, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
        bundle.world.load_state_dict(torch.load(path, map_location="cpu",
                                                weights_only=False)["world"])
        bundle.encoder.eval()
        bundle.world.eval()
        entry = {"checkpoint_sha256": _sha256(path)}
        for name, split, truth, episodes in (
                ("dev36", side["dev"], y["dev"], side["dev"]["episode"]),
                ("confirmation100", conf_split, conf_truth, conf_episodes)):
            gtr, rtr = generated(bundle, side["train"], settings)
            gdv, rdv = generated(bundle, split, settings)
            ra_tr = torch.cat((rtr[:, None].expand(-1, 17, -1), one_hot_actions(len(rtr))), -1)
            ra_dv = torch.cat((rdv[:, None].expand(-1, 17, -1), one_hot_actions(len(rdv))), -1)
            g = score(gtr, y["train"], gdv, truth, spec, args.device, order)
            c = controls.get(name)
            if c is None:
                c = controls[name] = score(ra_tr, y["train"], ra_dv, truth, spec, args.device, order)
            # paired interval on seed 0 scores
            gs = standardize(gtr, gdv)
            m, _, _ = fit_head(gs[0], y["train"], family="mlp128", objective="rank", seed=0,
                               spec=spec, device=args.device)
            cs = standardize(ra_tr, ra_dv)
            mc, _, _ = fit_head(cs[0], y["train"], family="mlp128", objective="rank", seed=0,
                                spec=spec, device=args.device)
            entry[name] = {"generated": g, "root_action_control": c,
                           "paired_generated_minus_control":
                               paired(scores_of(m, gs[1], args.device),
                                      scores_of(mc, cs[1], args.device), truth, episodes)}
            del gtr, gdv, rtr, rdv, ra_tr, ra_dv, gs, cs, m, mc
            torch.cuda.empty_cache()
            print(json.dumps({"stage": "scored", "arm": arm, "panel": name,
                              "generated": g["mean_safe"], "control": c["mean_safe"],
                              "frozen_auc_cost": g["frozen_auc_cost"],
                              "paired": entry[name]["paired_generated_minus_control"]}), flush=True)
        # batched: the whole held-out block at once exhausts a 6 GB card
        with torch.inference_mode():
            total, seen = 0.0, 0
            zs, acts = factual["z"][held], factual["actions"][held]
            for start in range(0, len(zs), 256):
                z = zs[start:start + 256].to(args.device)
                a = acts[start:start + 256].to(args.device)
                pairs = z.unsqueeze(2)
                pred = bundle.world.teacher(pairs, a).predicted
                total += float((pred.float() - pairs[:, 1:].float()).square().mean()) * len(z)
                seen += len(z)
            entry["factual_heldout_mse"] = total / seen
        print(json.dumps({"stage": "regression", "arm": arm,
                          "factual_heldout_mse": entry["factual_heldout_mse"]}), flush=True)
        report["arms"][arm] = entry
        del bundle
        torch.cuda.empty_cache()
    (args.out / "arms_eval.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "evaluation_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
