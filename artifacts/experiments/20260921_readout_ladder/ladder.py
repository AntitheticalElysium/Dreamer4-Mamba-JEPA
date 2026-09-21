"""Frozen Raw readout ladder: localize the H2 outcome failure without retraining a world.

The H2 gate showed the system cannot rank actions within a root, and the true-successor
substitution showed the reward head is no better when handed the REAL successor (regret 0.2879 vs
0.2781 generated). That rules out generated transition error as the main reward explanation, but a
real successor still traverses `observe_latent`, the agent readout, pooling and `model_body`, so
the failure sits somewhere in that shared route. This fits fresh heads on frozen features from
four points along it, on all-action roots disjoint from the gate's, and reads off where it breaks.

Feature families, materialized once from the frozen encoder and world:

  context_action   root agent features + one-hot action, NO successor. What is predictable before
                   seeing any successor at all.
  successor_z      the encoded real successor `z`. The representation itself, bypassing
                   `observe_latent` and the agent readout.
  real_features    `observe_latent(root, a, z_true)` features. The world's own readout of the REAL
                   successor.
  generated_features  `advance(root, a)` features. The deployed path.

Every family goes through an identical head -- Linear(D, 256) -> SwiGLU(256, 2.0) -> {reward bins,
continuation logit} -- at the capacity `Heads` actually uses, with identical initialization and
identical batch order, so a difference between families is about the features and not the fit.
The input adapter is declared: families differ in dimension and something must reconcile them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import twohot, _centers, _symlog
from d4mj.backbone import SwiGLU
from d4mj.config import config_from_dict
from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE
from d4mj.world_api import load_bundle

FAMILIES = ("context_action", "successor_cls", "successor_z", "real_features",
            "generated_features")


def rows_for(seeds, span, limit):
    """Fork rows whose seed lies in `seeds`, capped at `limit` roots."""
    wanted, out = set(seeds), []
    for path in sorted(FORK_STORE.glob("seed-*.pt")):
        if int(path.stem.split("-")[1]) not in wanted:
            continue
        for row in torch.load(path, map_location="cpu", weights_only=False):
            if len(row["frames"]) >= span:
                out.append(row)
                if len(out) >= limit:
                    return out
    return out


@torch.no_grad()
def materialize(bundle, rows, span, batch=8):
    """Every feature family plus the true outcomes, for all 17 actions of every root."""
    device, count = bundle.device, bundle.n_actions
    packs = {name: [] for name in FAMILIES}
    reward, terminated, seeds = [], [], []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        n = len(chunk)
        frames = torch.stack([r["frames"][-span:] for r in chunk]).to(device)
        past = torch.stack([r["led_to_action"][-span + 1:] for r in chunk]).to(device)
        successors = torch.stack([r["successors"] for r in chunk]).to(device)
        z = bundle.encoder(frames)
        state = bundle.world.teacher(z, past).state
        root_features = bundle.world.features(state)[:, -1, 0]            # [n, width]
        actions = torch.arange(count, device=device).repeat(n)[:, None]
        fan = bundle.repeat_state(state, count)
        advanced, generated = bundle.advance(fan, actions)
        # `export` returns the projected z the world transitions AND the unprojected CLS beside
        # it, in one pass. CLS is the rung upstream of the projector: if z fails where CLS
        # succeeds, the projector discarded it rather than the representation lacking it.
        z_true, cls_true, _ = bundle.encoder.export(successors.flatten(0, 1).unsqueeze(1))
        _, real = bundle.world.observe_latent(fan, actions, z_true)
        onehot = torch.nn.functional.one_hot(actions[:, 0], count).float()
        packs["context_action"].append(
            torch.cat((root_features.repeat_interleave(count, 0), onehot), -1).cpu())
        packs["successor_cls"].append(cls_true[:, 0].cpu())
        packs["successor_z"].append(z_true[:, 0, 0].cpu())
        packs["real_features"].append(real[:, -1, 0].cpu())
        packs["generated_features"].append(generated[:, -1, 0].cpu())
        reward.append(torch.stack([r["reward"] for r in chunk]).reshape(-1))
        terminated.append(torch.stack([r["terminated"] for r in chunk]).reshape(-1))
        seeds.append(torch.tensor([int(r["seed"]) for r in chunk]).repeat_interleave(count))
    data = {name: torch.cat(value).float() for name, value in packs.items()}
    data["reward"] = torch.cat(reward).float()
    data["terminated"] = torch.cat(terminated).bool()
    data["seed"] = torch.cat(seeds)
    data["roots"] = len(rows)
    return data


class Readout(nn.Module):
    """The capacity `Heads` uses, with a declared adapter so families share it exactly."""

    def __init__(self, width_in, width, bins):
        super().__init__()
        self.adapter = nn.Linear(width_in, width)
        self.body = SwiGLU(width, 2.0)
        self.reward = nn.Linear(width, bins)
        self.continuation = nn.Linear(width, 1)
        self.reward.weight.data.mul_(0.0)
        self.reward.bias.data.zero_()

    def forward(self, x):
        h = self.body(self.adapter(x))
        return self.reward(h), self.continuation(h)


def fit(features, reward, alive, centers, *, steps, seed, device, width, bins, holdout,
        batch=512, check_every=100):
    """Fit with model selection on an inner slice of the FIT roots -- never on the judgement roots.

    Without this the ladder cannot answer its own question. At 600 roots every family, including
    `context_action` which carries no successor information at all, reached a within-root
    association above 0.9 on the data it was fitted on and near zero on held-out roots. A DEV
    failure measured off the last update is therefore consistent with the features carrying the
    outcome perfectly and the head having memorized, which is exactly the confound the ladder
    exists to remove. The selection slice is carved off by ROOT, so the 17 rows of a root never
    straddle it, and it comes out of the fit partition, so the judgement roots stay untouched.
    """
    torch.manual_seed(seed)
    model = Readout(features.shape[-1], width, bins).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    generator = torch.Generator().manual_seed(seed + 1)
    target = twohot(_symlog(reward), centers.cpu()).to(device)
    features, alive = features.to(device), alive.to(device).float()
    inner = torch.where(holdout.to(device))[0]
    outer = torch.where(~holdout.to(device))[0]

    def objective(index):
        logits, continuation = model(features[index])
        return (-(target[index] * logits.log_softmax(-1)).sum(-1).mean()
                + nn.functional.binary_cross_entropy_with_logits(continuation[:, 0], alive[index]))

    @torch.no_grad()
    def selection():
        model.eval()
        total = sum(float(objective(inner[start:start + 4096])) * len(inner[start:start + 4096])
                    for start in range(0, len(inner), 4096))
        model.train()
        return total / len(inner)

    best, chosen, trace = float("inf"), 0, []
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for step in range(steps):
        pick = outer[torch.randint(len(outer), (batch,), generator=generator).to(device)]
        loss = objective(pick)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        trace.append(float(loss.detach()))
        if (step + 1) % check_every == 0:
            score = selection()
            if score < best:
                best, chosen = score, step + 1
                state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(state)
    # Carried on the model so a family that simply failed to optimize is distinguishable in the
    # evidence from one whose features do not carry the outcome.
    model.trace = {"steps": steps, "selected_step": chosen, "selection_loss": best,
                   "fit_first_100": float(np.mean(trace[:100])),
                   "fit_last_100": float(np.mean(trace[-100:]))}
    return model.eval()


@torch.no_grad()
def predict(model, features, centers, device, batch=4096):
    reward, death = [], []
    for start in range(0, len(features), batch):
        logits, continuation = model(features[start:start + batch].to(device))
        mean = (logits.softmax(-1) * centers).sum(-1)
        reward.append((mean.sign() * torch.expm1(mean.abs())).cpu())
        death.append((1.0 - continuation[:, 0].sigmoid()).cpu())
    return torch.cat(reward), torch.cat(death)


def paired(left, right, clusters, *, draws, seed):
    """Episode-seed-clustered bootstrap of mean(left) - mean(right)."""
    groups = [torch.where(clusters == k)[0] for k in clusters.unique(sorted=True)]
    if len(groups) < 2:
        return {"difference": None, "interval": None, "excludes_zero": False}
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(draws):
        pick = torch.cat([groups[i] for i in
                          torch.randint(len(groups), (len(groups),), generator=generator)])
        samples.append(float(left[pick].mean() - right[pick].mean()))
    low, high = torch.tensor(samples).quantile(torch.tensor([.025, .975])).tolist()
    return {"difference": float(left.mean() - right.mean()), "interval": [low, high],
            "excludes_zero": bool(low > 0 or high < 0)}


def decisions(truth_r, truth_d, pred_r, pred_d, seeds, count, *, draws, seed):
    """Within-root decision quality against the action-marginal control."""
    rows = len(truth_r) // count
    tr = truth_r.reshape(rows, count)
    td = truth_d.reshape(rows, count).float()
    pr = pred_r.reshape(rows, count)
    pd = pred_d.reshape(rows, count)
    root_seed = seeds.reshape(rows, count)[:, 0]
    r_varies, d_varies = tr.amax(1) > tr.amin(1), td.amax(1) > td.amin(1)

    def regret(truth, score, maximize):
        chosen = (score.argmax(1) if maximize else score.argmin(1))[:, None]
        best = truth.amax(1, keepdim=True) if maximize else truth.amin(1, keepdim=True)
        return (best - truth.gather(1, chosen)).squeeze(1).abs()

    out = {"roots": rows, "reward_opportunity_roots": int(r_varies.sum()),
           "terminal_opportunity_roots": int(d_varies.sum())}
    # The per-root vectors, kept so a family can be contrasted against ANOTHER FAMILY on the same
    # roots and not only against the action-marginal. The opportunity masks come from the truth
    # alone, so they are identical across families and the vectors line up row for row.
    out["_vectors"] = vectors = {}
    if int(r_varies.sum()) >= 24:
        t, p = tr[r_varies], pr[r_varies]
        marginal = t.mean(0, keepdim=True).expand_as(t)
        model, blind = regret(t, p, True), regret(t, marginal, True)
        # within-root centered association: does the score track the truth AFTER removing the
        # per-root mean? A model that only knows which actions are good on average scores zero.
        tc, pc = t - t.mean(1, keepdim=True), p - p.mean(1, keepdim=True)
        assoc = float((tc * pc).sum() / (tc.norm() * pc.norm()).clamp_min(1e-9))
        out["reward"] = {"regret": float(model.mean()), "marginal_regret": float(blind.mean()),
                         "within_root_association": assoc,
                         "vs_marginal": paired(blind, model, root_seed[r_varies],
                                               draws=draws, seed=seed + 1)}
        out["reward"]["beats_marginal"] = bool(
            (out["reward"]["vs_marginal"].get("difference") or 0) > 0
            and out["reward"]["vs_marginal"]["excludes_zero"])
        vectors["reward"] = (model, root_seed[r_varies])
    else:
        out["reward"] = {"status": "insufficient_coverage"}
    if int(d_varies.sum()) >= 24:
        t, p = td[d_varies], pd[d_varies]
        marginal = t.mean(0, keepdim=True).expand_as(t)
        safe = 1.0 - regret(t, p, False)
        blind = 1.0 - regret(t, marginal, False)
        tc, pc = t - t.mean(1, keepdim=True), p - p.mean(1, keepdim=True)
        assoc = float((tc * pc).sum() / (tc.norm() * pc.norm()).clamp_min(1e-9))
        out["terminal"] = {"safe_choice": float(safe.mean()),
                           "marginal_safe_choice": float(blind.mean()),
                           "within_root_association": assoc,
                           "vs_marginal": paired(safe, blind, root_seed[d_varies],
                                                 draws=draws, seed=seed + 2)}
        out["terminal"]["beats_marginal"] = bool(
            (out["terminal"]["vs_marginal"].get("difference") or 0) > 0
            and out["terminal"]["vs_marginal"]["excludes_zero"])
        vectors["terminal"] = (safe, root_seed[d_varies])
    else:
        out["terminal"] = {"status": "insufficient_coverage"}
    return out


def verdict(matrix):
    """The declared decision rules, applied to the cross-matrix.

    `successor_cls` is deliberately absent here. It was added as a supplementary rung after the
    rules were declared, and a rung added later must not be able to move a predeclared decision.
    """
    def ok(family, fitted, split):
        cell = matrix.get(family, {}).get(fitted, {}).get(split, {})
        return (cell.get("reward", {}).get("beats_marginal", False),
                cell.get("terminal", {}).get("beats_marginal", False))
    real_r, real_t = ok("real_features", "real", "dev")
    gen_r, gen_t = ok("generated_features", "generated", "dev")
    z_r, z_t = ok("successor_z", "real", "dev")
    ctx_r, ctx_t = ok("context_action", "real", "dev")
    real_any, gen_any, z_any = real_r or real_t, gen_r or gen_t, z_r or z_t
    if real_any and gen_any:
        call = "bridge_head_data_or_objective"
        why = ("fresh heads on all-action data succeed from BOTH real and generated features, so "
               "the features carry the information and the bridge heads' data/objective is the "
               "problem -- consistent with only 25% of their supervision coming from generated "
               "suffixes, all on logged actions")
    elif real_any and not gen_any:
        call = "transition_or_generated_feature"
        why = "real-successor features succeed where generated ones fail: the generated path is at fault"
    elif not real_any and z_any:
        call = "observe_latent_or_agent_readout_bottleneck"
        why = ("the encoded successor z carries the information but the world's own readout of it "
               "does not: the bottleneck is observe_latent / the agent readout")
    elif not (real_any or gen_any or z_any or ctx_r or ctx_t):
        call = "representation_or_context_deficiency"
        why = "no latent route succeeds; the representation or the context is deficient"
    else:
        call = "mixed"
        why = "the pattern does not match a single declared rule; read the matrix"
    return {"call": call, "reasoning": why,
            "note": "a route that succeeds on TRAIN but fails on DEV is a capacity/generalization "
                    "problem and must not be integrated; see train_dev_gap"}


def _gap(cell):
    """TRAIN minus DEV on the headline metrics. Reward regret is lower-is-better, so a NEGATIVE
    reward gap means TRAIN is better; safe-choice is higher-is-better, so a POSITIVE terminal gap
    means TRAIN is better. Either way a large magnitude is memorization, not a usable route."""
    out = {}
    for key, metric in (("reward", "regret"), ("terminal", "safe_choice")):
        train, dev = cell["train"].get(key, {}), cell["dev"].get(key, {})
        if metric in train and metric in dev:
            out[key] = float(train[metric] - dev[metric])
    return out


# Which fitted head may be evaluated on which family. Only `real_features` and `generated_features`
# share an input dimension (both are `width`), so only that pair can cross; `context_action`
# (width + n_actions) and `successor_z` (d_bottleneck) are diagonal by construction.
CROSS = {"context_action": ("context_action",),
         "successor_cls": ("successor_cls",),
         "successor_z": ("successor_z",),
         "real_features": ("real_features", "generated_features"),
         "generated_features": ("generated_features", "real_features")}
# How the matrix labels the head that produced a cell, so `verdict` can read it.
LABEL = {"context_action": "real", "successor_cls": "real", "successor_z": "real",
         "real_features": "real", "generated_features": "generated"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--train-roots", type=int, default=4000)
    parser.add_argument("--dev-roots", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from d4mj.experiments import _load_bridge_parent
    from d4mj.lewm_config import window_layout

    checkpoint = args.checkpoint or args.run / "bridge/step-002000.pt"
    bundle, heads, payload = _load_bridge_parent(checkpoint)
    # Exactly the gate's frozen posture: exported encoder, streaming world, fixed BatchNorm.
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    heads.eval()
    config = bundle.config
    offsets = window_layout(config.joint)[0]
    span = offsets[-1] + 1
    if offsets != tuple(range(span)):
        raise SystemExit("fork windows assume a consecutive encoded window")
    device, count = bundle.device, bundle.n_actions
    width, bins = config.dynamics.width, config.agent.bins
    # The DEPLOYED reward grid, taken off the trained heads rather than rebuilt, so a fresh head
    # is scored on the same bins the bridge head was.
    centers = heads.centers.detach().clone().to(device)

    partition = json.loads(args.partition.read_text())
    splits = {}
    for name, key, limit in (("train", "fit_train", args.train_roots),
                             ("dev", "fit_dev", args.dev_roots)):
        rows = rows_for(partition[key]["seeds"], span, limit)
        splits[name] = materialize(bundle, rows, span, batch=16)
        print(json.dumps({"stage": "materialized", "split": name,
                          "roots": splits[name]["roots"],
                          "rows": int(len(splits[name]["reward"])),
                          "seconds": round(time.time() - started, 1)}), flush=True)
    if splits["train"]["roots"] < 64 or splits["dev"]["roots"] < 64:
        raise SystemExit("the sealed partition yielded too few usable roots to fit or judge")
    seeds_train = set(splits["train"]["seed"].tolist())
    seeds_dev = set(splits["dev"]["seed"].tolist())
    if seeds_train & seeds_dev:
        raise SystemExit("fit and judgement roots share an episode seed")

    del bundle, heads, payload
    torch.cuda.empty_cache()

    # The selection slice, carved off the FIT roots by root so the 17 rows of a root stay together,
    # and identical across families so a difference between them is about features, not the split.
    fit_roots = splits["train"]["roots"]
    inner = torch.zeros(fit_roots, dtype=torch.bool)
    inner[torch.randperm(fit_roots, generator=torch.Generator().manual_seed(args.seed + 3))
          [: max(1, int(round(0.15 * fit_roots)))]] = True
    holdout = inner.repeat_interleave(count)

    # `context_action` is FAMILIES[0], so the no-successor control is in hand before any
    # successor family is scored.
    control = {}
    matrix = {family: {} for family in FAMILIES}
    for family in FAMILIES:
        model = fit(splits["train"][family], splits["train"]["reward"],
                    ~splits["train"]["terminated"], centers, steps=args.steps, seed=args.seed,
                    device=device, width=width, bins=bins, holdout=holdout)
        for target in CROSS[family]:
            cell = {"fitted_on": family, "evaluated_on": target}
            for split in ("train", "dev"):
                data = splits[split]
                predicted_reward, predicted_death = predict(model, data[target], centers, device)
                cell[split] = decisions(data["reward"], data["terminated"], predicted_reward,
                                        predicted_death, data["seed"], count,
                                        draws=args.draws, seed=args.seed + 7)
            if target == "context_action":
                control = {(split, key): value for split in ("train", "dev")
                           for key, value in cell[split]["_vectors"].items()}
            # Against the action-marginal, `context_action` answers "does this beat ignoring the
            # state?". Against `context_action`, a successor family answers the sharper question
            # the ladder exists for: does SEEING the successor add anything over root and action?
            for split in ("train", "dev"):
                for key, higher_is_better in (("reward", False), ("terminal", True)):
                    mine = cell[split]["_vectors"].get(key)
                    theirs = control.get((split, key))
                    if mine is None or theirs is None or target == "context_action":
                        continue
                    left, right = (mine[0], theirs[0]) if higher_is_better else (theirs[0], mine[0])
                    cell[split][key]["vs_context_action"] = paired(
                        left, right, mine[1], draws=args.draws, seed=args.seed + 11)
                del cell[split]["_vectors"]
            cell["train_dev_gap"] = _gap(cell)
            cell["fit_trace"] = model.trace
            matrix[target][LABEL[family]] = cell
            print(json.dumps({"stage": "scored", "fitted_on": family, "evaluated_on": target,
                              "dev_reward_beats": cell["dev"]["reward"].get("beats_marginal"),
                              "dev_terminal_beats": cell["dev"]["terminal"].get("beats_marginal"),
                              "seconds": round(time.time() - started, 1)}), flush=True)
        del model
        torch.cuda.empty_cache()

    report = {"schema": "d4mj_readout_ladder_v1",
              "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              "partition_sha256": _sha256(args.partition), "script_sha256": _sha256(Path(__file__)),
              "head": "Linear(D, 256) -> SwiGLU(256, 2.0) -> {reward bins, continuation}; identical "
                      "initialization and identical batch order across families. The input adapter "
                      "is declared: families differ in dimension and something must reconcile them.",
              "steps": args.steps, "draws": args.draws, "seed": args.seed,
              "selection": "model selection every 100 updates on 15% of the FIT roots, held out by "
                           "root; the judgement roots are never used to select anything",
              "roots": {name: splits[name]["roots"] for name in splits},
              "matrix_layout": "matrix[evaluated_on][fitted_on_label][split]",
              "gap_note": "train_dev_gap is TRAIN minus DEV; reward regret is lower-is-better and "
                          "safe_choice higher-is-better, so read the sign per metric",
              "matrix": matrix}
    report["verdict"] = verdict(matrix)
    (args.out / "ladder.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "ladder_complete", "seconds": round(time.time() - started, 1),
                      **report["verdict"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
