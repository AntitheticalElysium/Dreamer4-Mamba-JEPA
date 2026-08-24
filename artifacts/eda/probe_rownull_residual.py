"""Where does the action-specific structure live, and can it correct the successor?

The corrected gate leaves a real but smaller drop: centred and capacity-matched,
hidden 0.8102 against pre_tanh 0.7354. The terminal projection W is 32 x 256, so per
token it can see a 32-dimensional row space and discards a 224-dimensional null space.
This asks where the extractable structure sits relative to that split, and then whether
it is successor information the trained readout simply fails to use.

Part 2, damage probes, all capacity-matched to width 1024 and all on centred tokens:

  full          every centred hidden token, orthonormal 8192 -> 1024
  row           the 32 directions W can see, 32 tokens x 32 = 1024, identity
  null          the 224 it discards, orthonormal 7168 -> 1024
  row/null shuffled   candidate actions permuted within each root
  token_mean    tokens averaged first, 256 wide -- is the signal distributed?

Part 3 stops asking about damage and asks about the target. The residual

    e = (z* - mean_a z*) - (zhat - mean_a zhat)

is exactly the action-conditioned successor error the trained head leaves behind. A
diagnostic extractor is fitted to predict e from each view, its held-out prediction is
added back to the model's own, and the corrected successor is rescored. A positive
held-out correction is the evidence the decoder sweep could not produce: hidden
carrying successor information the readout does not use.

  same_token    hidden token i -> residual slice i, a per-token map
  all_token     all tokens -> all 1024, so cross-token routing is available

Part 4 already ran separately: across 5,146 multi-action equivalence classes, identical
latent successors never disagree on the damage label, so the target is sufficient and
tokenizer aliasing is excluded.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_phase1b_fork import predict_branches
from probe_action_matching import classes_of
from probe_hidden_ceiling import cache_hidden
from probe_hidden_forensics import fit_and_score, fit_residual, row_null_bases, standardise
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import MixerWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config

DEVICE = "cuda"
SUFFIX = "abm0"
WIDTH = 1024


def ortho(rows_in, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.linalg.qr(torch.randn(rows_in, WIDTH, generator=generator))[0].half()


def main() -> None:
    rows = load_forkset(HERE / f"forkset_{SUFFIX}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    masks = tuple(torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1].float()

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = MixerWorld(config).to(DEVICE)
    load(HERE / f"phase1b_{SUFFIX}_n64" / "world_020000.pt", config, part0=world)
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)

    led = fork_actions(rows)
    raw = cache_hidden(world, config, history, led)
    with torch.no_grad():
        hidden = torch.cat([world.mix_norm(raw[lo:lo+64].to(DEVICE).float()).half().cpu()
                            for lo in range(0, len(raw), 64)])
    del raw
    predicted = predict_branches(world, config, history, led).float()
    print(f"hidden {tuple(hidden.shape)}  predicted {tuple(predicted.shape)}", flush=True)

    hc = (hidden.float() - hidden.float().mean(1, keepdim=True))          # centred tokens
    v_row, v_null = row_null_bases(world.readout[2].weight.detach().cpu())
    views = {
        "full": (hc.reshape(*hc.shape[:2], -1).half() @ ortho(hc.shape[2] * hc.shape[3], 1)),
        "row": (hc @ v_row).reshape(*hc.shape[:2], -1).half(),
        "null": ((hc @ v_null).reshape(*hc.shape[:2], -1).half() @ ortho(hc.shape[2] * 224, 2)),
        "token_mean": hc.mean(2).half(),
    }
    del hc
    result = {"arm": SUFFIX, "damage": {}, "residual": {}}

    print("\nPart 2: damage probes, capacity-matched, centred")
    generator = torch.Generator().manual_seed(9)
    for name, x in views.items():
        xs = standardise(x, masks[0])
        result["damage"][name] = fit_and_score(xs, labels, masks, seed=11)
        r = result["damage"][name]
        print(f"  {name:<16} AUC {r['test_auc']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
              flush=True)
        if name in ("row", "null"):
            order = torch.stack([torch.randperm(17, generator=generator)
                                 for _ in range(len(x))])
            shuffled = torch.gather(x, 1, order[..., None].expand_as(x))
            key = f"{name}_shuffled"
            result["damage"][key] = fit_and_score(standardise(shuffled, masks[0]),
                                                  labels, masks, seed=11)
            r = result["damage"][key]
            print(f"  {key:<16} AUC {r['test_auc']:.4f} [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
                  flush=True)

    # ---------------------------------------------------------------- Part 3
    target = ((branch - branch.mean(1, keepdim=True))
              - (predicted - predicted.mean(1, keepdim=True)))
    test = masks[2]
    print(f"\nPart 3: residual e, energy {float(target[test].pow(2).mean()):.5f} "
          f"against action effect "
          f"{float((branch[test]-branch[test].mean(1,keepdim=True)).pow(2).mean()):.5f}")

    plans = {"full": views["full"], "row": views["row"], "null": views["null"]}
    for name, x in plans.items():
        xs = standardise(x, masks[0])
        pred_e, tune_loss = fit_residual(xs, target, masks, seed=13)
        keep = 1 - float((pred_e - target[test]).pow(2).sum() / target[test].pow(2).sum())
        corrected = predicted[test] + pred_e
        result["residual"][name] = {"residual_r2": keep, "tune_loss": tune_loss,
                                    **rescore(corrected, branch[test], labels[test.numpy()])}
        r = result["residual"][name]
        print(f"  {name:<12} residual R2 {keep:+.4f}   corrected NSE {r['nse']:.4f}  "
              f"cosine {r['cosine']:.4f}  geometry {r['geometry']:.4f}", flush=True)

    # same_token: hidden token i -> residual slice i, no cross-token routing available.
    # Linear broadcasts over leading dims, so 4-D in gives a per-token map for free.
    tokens = standardise((hidden.float() - hidden.float().mean(1, keepdim=True)).half(),
                         masks[0])
    sliced = target.view(*target.shape[:2], config.n_spatial, config.d_spatial)
    pred_e, tune_loss = fit_residual(tokens, sliced, masks, seed=13)
    pred_e = pred_e.reshape(int(test.sum()), 17, -1)
    keep = 1 - float((pred_e - target[test]).pow(2).sum() / target[test].pow(2).sum())
    corrected = predicted[test] + pred_e
    result["residual"]["same_token"] = {"residual_r2": keep, "tune_loss": tune_loss,
                                        **rescore(corrected, branch[test], labels[test.numpy()])}
    r = result["residual"]["same_token"]
    print(f"  {'same_token':<12} residual R2 {keep:+.4f}   corrected NSE {r['nse']:.4f}  "
          f"cosine {r['cosine']:.4f}  geometry {r['geometry']:.4f}", flush=True)

    # baseline: the uncorrected model
    result["residual"]["uncorrected"] = rescore(predicted[test], branch[test],
                                                labels[test.numpy()])
    r = result["residual"]["uncorrected"]
    print(f"  {'uncorrected':<12} {'':>16}   NSE {r['nse']:.4f}  cosine {r['cosine']:.4f}  "
          f"geometry {r['geometry']:.4f}")

    (HERE / f"rownull_residual_{SUFFIX}.json").write_text(json.dumps(result, indent=2))


def rescore(pred, truth, labels_test):
    hat = pred - pred.mean(1, keepdim=True)
    eff = truth - truth.mean(1, keepdim=True)
    nse = ((hat - eff).pow(2).sum(-1) / eff.pow(2).sum(-1).clamp(min=1e-12)).mean()
    cos = torch.nn.functional.cosine_similarity(hat.reshape(-1, hat.shape[-1]),
                                                eff.reshape(-1, eff.shape[-1]), dim=1).mean()
    gram = torch.cdist(eff, eff).pow(2).numpy()
    pgram = torch.cdist(hat, hat).pow(2).numpy()
    cross = torch.cdist(hat, eff).pow(2).numpy()
    hits, geo = [], []
    for i in range(len(cross)):
        lab = classes_of(gram[i])
        if lab.max() < 1:
            continue
        best = cross[i].argmin(1)
        hits.append(np.mean([lab[best[a]] == lab[a] for a in range(17)]))
        u = np.triu_indices(17, 1)
        if gram[i][u].std() > 0 and pgram[i][u].std() > 0:
            geo.append(np.corrcoef(gram[i][u], pgram[i][u])[0, 1])
    return {"nse": float(nse), "cosine": float(cos), "retrieval": float(np.mean(hits)),
            "geometry": float(np.mean(geo))}


if __name__ == "__main__":
    main()
