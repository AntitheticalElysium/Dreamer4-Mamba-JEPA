"""Step 1: verify the audit independently, and decompose the effect by semantic direction.

An audit reported that B improved the counterfactual LATENT objective substantially -- branch
MSE .0277 -> .0132, effect R^2 .803 -> .858 -- while safety choice did not move. If that holds,
my "trending negative / centroid collapse" reading is wrong and the result is far more
interesting: the world learns action effects better in Raw-z geometry without those effects
buying the semantics planning needs.

So this re-measures the audit's claims from the checkpoints rather than accepting them, and adds
the decomposition that tells us WHERE B's improved effect fidelity went:

  effect        all-17 branch MSE, effect R^2 and cosine on the fork pool, plus action retrieval
                (does the predicted effect identify WHICH action produced it?)
  derangement   20 draws, not one -- the ladder already showed a single draw is noise and the
                evaluator repeated that mistake
  semantic      per-target decoders FIT ON REAL successors and applied to generated states, for
                every outcome the gate scores. This is the gate's actual transfer direction, and
                it says whether B's gain landed on death/damage/inventory/tile or nowhere useful

IN-SAMPLE WARNING: A' and B trained on the whole fork pool, so the effect metrics below are
in-sample for them and out-of-sample for A. They are reported for comparability with the audit
and are NOT evidence of generalization. The from-scratch run must hold out fork seeds up front.
"""

import argparse
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
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256
from d4mj.m03.gate import M03Settings, load_m03_bundle
from readout import Fit, fit_head, scores_of, standardize, summarize
from evaluate_arms import RAW, DATASET, SIDECAR, generated, confirmation_index
from phase4 import derangement

OUTCOMES = ("death", "damage", "reward_positive", "achievement_event",
            "inventory_changed", "tile_changed")
ARMS = ("A_control", "Ap_data", "B_sibling")
SEEDS = (0, 1, 2)


@torch.inference_mode()
def fork_effect(bundle, fork, batch=16, limit=None):
    """Predict all 17 branches for each fork root; score effect fidelity and action retrieval."""
    n = len(fork["z_branch"]) if limit is None else min(limit, len(fork["z_branch"]))
    device = bundle.device
    acts = torch.arange(17, device=device)
    se, n_el, preds, trues, roots = 0.0, 0, [], [], []
    for start in range(0, n, batch):
        end = min(n, start + batch)
        m = end - start
        zh = fork["z_history"][start:end].to(device)
        past = fork["past_actions"][start:end].to(device)
        state = bundle.prefill(zh.unsqueeze(2), past)
        branches = bundle.repeat_state(state, 17)
        predicted, _ = bundle.advance(branches, acts.repeat(m)[:, None])
        got = predicted.latent[:, 0, 0].reshape(m, 17, -1).cpu()
        tgt = fork["z_branch"][start:end]
        root = fork["z_history"][start:end, -1]
        se += float((got.float() - tgt.float()).square().sum())
        n_el += got.numel()
        preds.append(got.float() - root[:, None].float())      # predicted EFFECT
        trues.append(tgt.float() - root[:, None].float())      # true EFFECT
        roots.append(root)
    pred, true = torch.cat(preds), torch.cat(trues)
    resid = (pred - true).square().sum()
    total = (true - true.mean((0, 1), keepdim=True)).square().sum()
    cos = torch.nn.functional.cosine_similarity(pred.flatten(0, 1), true.flatten(0, 1), dim=-1)
    # action retrieval: does the predicted effect match its OWN action's true effect best?
    hit = 0
    for i in range(len(pred)):
        d = torch.cdist(pred[i], true[i])
        hit += int((d.argmin(1) == torch.arange(17)).sum())
    return {"roots": n, "branch_mse": se / n_el, "effect_r2": float(1 - resid / total),
            "effect_cosine": float(cos.mean()), "action_retrieval": hit / (n * 17)}


def semantic(real_tr, real_dv, gen_dv, ytr, ydv, spec, device):
    """Decoders FIT ON REAL successors, applied to real (ceiling) and generated (transfer)."""
    out = {}
    xtr, xdv_real = standardize(real_tr, real_dv)
    _, xdv_gen = standardize(real_tr, gen_dv)
    for index, name in enumerate(OUTCOMES):
        truth_tr = ytr[..., index:index + 1]
        if truth_tr.sum() == 0 or (1 - truth_tr).sum() == 0:
            continue
        aucs = {"real": [], "generated": []}
        for seed in SEEDS:
            model, _, _ = fit_head(xtr, truth_tr, family="mlp128", objective="bce6",
                                   seed=seed, spec=spec, device=device)
            for key, xs in (("real", xdv_real), ("generated", xdv_gen)):
                with torch.inference_mode():
                    p = torch.cat([model(xs[i:i + 128].to(device))[..., 0].cpu()
                                   for i in range(0, len(xs), 128)]).reshape(-1)
                t = ydv[..., index].reshape(-1).bool()
                if t.all() or (~t).all():
                    continue
                aucs[key].append(float((p[t][:, None] > p[~t][None, :]).float().mean()))
        if aucs["real"]:
            out[name] = {"real_auc": round(float(np.mean(aucs["real"])), 4),
                         "generated_auc": round(float(np.mean(aucs["generated"])), 4)}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--cache", type=Path, default=HERE / "cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fork-limit", type=int, default=4096)
    parser.add_argument("--draws", type=int, default=20)
    args = parser.parse_args(argv)

    settings, spec = M03Settings(), Fit(steps=2000)
    side = torch.load(SIDECAR, map_location="cpu", weights_only=False)["splits"]
    y = {s: side[s]["outcomes"].float() for s in ("train", "dev")}
    fork = torch.load(args.cache / "fork.pt", map_location="cpu", weights_only=False)
    gen_rng = torch.Generator().manual_seed(4242)
    orders = [derangement(17, gen_rng) for _ in range(args.draws)]

    report = {"schema": "d4mj_cf_decompose_v1",
              "in_sample_warning": "A' and B trained on the ENTIRE fork pool, so effect metrics are "
                                   "in-sample for them and out-of-sample for A. Not generalization.",
              "arms": {}}
    real = {}
    for arm in ARMS:
        path = args.runs / arm / "world_010000.pt"
        bundle, _, _ = load_m03_bundle(RAW, device=args.device, dataset_sha256=_sha256(DATASET))
        bundle.world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"])
        bundle.encoder.eval(); bundle.world.eval()
        entry = {"effect": fork_effect(bundle, fork, limit=args.fork_limit)}
        print(json.dumps({"stage": "effect", "arm": arm, **entry["effect"]}), flush=True)

        gtr, _ = generated(bundle, side["train"], settings)
        gdv, _ = generated(bundle, side["dev"], settings)
        if not real:
            # encoder-only, identical for every arm
            real["train"] = side["train"]["successors"]
            real["dev"] = side["dev"]["successors"]
            enc = bundle.encoder
            with torch.inference_mode():
                def enc_all(x):
                    n, a = x.shape[:2]
                    flat = x.reshape(-1, *x.shape[2:])
                    outs = []
                    for i in range(0, len(flat), 256):
                        z, _ = enc.projected_and_cls(flat[i:i + 256].unsqueeze(1).to(bundle.device))
                        outs.append(z[:, 0, 0].cpu())
                    return torch.cat(outs).reshape(n, a, -1)
                real["train"] = enc_all(real["train"])
                real["dev"] = enc_all(real["dev"])
        entry["semantic_realfit_to_generated"] = semantic(
            real["train"], real["dev"], gdv, y["train"], y["dev"], spec, args.device)
        print(json.dumps({"stage": "semantic", "arm": arm,
                          **entry["semantic_realfit_to_generated"]}), flush=True)

        # 20-draw frozen-head derangement on the generated states
        xtr, xdv = standardize(gtr, gdv)
        intact, perm = [], []
        for seed in SEEDS:
            model, _, _ = fit_head(xtr, y["train"], family="mlp128", objective="rank",
                                   seed=seed, spec=spec, device=args.device)
            intact.append(summarize(scores_of(model, xdv, args.device), y["dev"])["within_root_auc"])
            perm.append(float(np.mean([summarize(scores_of(model, xdv[:, o], args.device),
                                                 y["dev"])["within_root_auc"] for o in orders])))
        entry["derangement_20"] = {"intact_auc": round(float(np.mean(intact)), 4),
                                   "permuted_auc": round(float(np.mean(perm)), 4),
                                   "frozen_auc_cost": round(float(np.mean(intact)) - float(np.mean(perm)), 4)}
        print(json.dumps({"stage": "derangement20", "arm": arm, **entry["derangement_20"]}), flush=True)
        report["arms"][arm] = entry
        del bundle
        torch.cuda.empty_cache()
    (args.out / "decompose.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "decompose_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
