"""Confirmation run: the trained bridge head and an exact-capacity refit, on the same untouched roots.

The ladder established that safety information is ACCESSIBLE from real-successor features. It did
not establish the separate claim that the TRAINED bridge head fails to read it, because two
confounds stood in the way: the fresh head had a trainable input adapter and a one-step objective
rather than the deployed `Heads.model_body + reward/continuation`, and it was scored on a
different root population (3,369 ladder roots) than the trained head (512 gate roots).

This run removes both. Every head below is scored on ONE population -- the 405 fork seeds left
unallocated by the sealed partition, which no head here was fitted or selected on -- and the
deployed outcome path is reproduced exactly, with no adapter, beside the adapter head kept as a
capacity-positive control.

It also adds the `generated_z` rung, reading `advanced.latent` BEFORE `world.features`, which the
ladder could not distinguish from a corrupted readout of a usable latent.

Decisive branches, all on identical roots:

  exact fresh head succeeds, trained head fails   -> bridge training data/objective
  adapter succeeds, exact fails                   -> readout capacity/interface
  generated_z fails                               -> the transition's latent is the problem
  generated_z succeeds, generated_features fails  -> the generated agent readout is the problem
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.backbone import SwiGLU
from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE, _fork_readouts

from ladder import Readout, _gap, decisions, fit, paired, predict, verdict  # noqa: E402

# `generated_z` is the new rung: the transition's own latent, before the agent readout touches it.
FAMILIES = ("context_action", "successor_cls", "successor_z", "generated_z",
            "real_features", "generated_features")
# Only these are `width`-dimensional, so only these can go through the deployed head unmodified.
EXACT_OK = ("real_features", "generated_features")
CROSS = {name: (name,) for name in FAMILIES}
CROSS["real_features"] = ("real_features", "generated_features")
CROSS["generated_features"] = ("generated_features", "real_features")


class Exact(nn.Module):
    """The deployed outcome path, reproduced with NO adapter: `Heads.model_body` and its two heads.

    `Heads.forward` pools the token axis with `agent.mean(dim=2)`; the fork features have a single
    token, so the `[:, -1, 0]` taken at materialization already IS that pooled vector and no
    pooling difference is introduced. The deployed reward head emits `mtp_leads * bins` and is
    trained across 8 leads; the fork roots carry one-step outcomes only, so this fits and reads
    lead 0. That makes the exact head's job EASIER than the deployed one's, which is the right
    direction for a control meant to rule out capacity.
    """

    def __init__(self, width_in, width, bins):
        super().__init__()
        if width_in != width:
            raise ValueError("the deployed head takes agent features unmodified; no adapter here")
        self.body = SwiGLU(width, 2.0)
        self.reward = nn.Linear(width, bins)
        self.continuation = nn.Linear(width, 1)
        self.reward.weight.data.mul_(0.0)
        self.reward.bias.data.zero_()

    def forward(self, x):
        h = self.body(x)
        return self.reward(h), self.continuation(h)


def seeds_for(partition, store):
    """The three disjoint groups, with `unallocated` derived: the partition recorded only its count."""
    allocated = set()
    for key in ("reserved_for_gate", "fit_train", "fit_dev"):
        allocated |= set(partition[key]["seeds"])
    every = sorted(int(p.stem.split("-")[1]) for p in Path(store).glob("seed-*.pt"))
    unallocated = sorted(set(every) - allocated)
    if len(unallocated) != partition["unallocated"]["count"]:
        raise SystemExit("derived unallocated seeds disagree with the sealed partition count")
    return sorted(partition["fit_train"]["seeds"]), unallocated


def pack(rows, span):
    """The `fork_population` layout, so the TRAINED head is read by the closure's own code."""
    stack = lambda key: torch.stack([row[key] for row in rows])
    return {"frames": torch.stack([row["frames"][-span:] for row in rows]),
            "past_actions": torch.stack([row["led_to_action"][-span + 1:] for row in rows]),
            "successors": stack("successors"), "reward": stack("reward").float(),
            "terminated": stack("terminated").bool(), "roots": len(rows)}


@torch.no_grad()
def materialize(bundle, rows, span, batch=16):
    """The ladder's families plus `generated_z`, and a digest binding the exact rows measured."""
    device, count = bundle.device, bundle.n_actions
    packs = {name: [] for name in FAMILIES}
    reward, terminated, seeds, identity = [], [], [], hashlib.sha256()
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        n = len(chunk)
        for row in chunk:
            identity.update(repr((int(row["seed"]), int(row["step"]))).encode())
        frames = torch.stack([r["frames"][-span:] for r in chunk]).to(device)
        past = torch.stack([r["led_to_action"][-span + 1:] for r in chunk]).to(device)
        successors = torch.stack([r["successors"] for r in chunk]).to(device)
        z = bundle.encoder(frames)
        state = bundle.world.teacher(z, past).state
        root_features = bundle.world.features(state)[:, -1, 0]
        actions = torch.arange(count, device=device).repeat(n)[:, None]
        fan = bundle.repeat_state(state, count)
        advanced, generated = bundle.advance(fan, actions)
        z_true, cls_true, _ = bundle.encoder.export(successors.flatten(0, 1).unsqueeze(1))
        _, real = bundle.world.observe_latent(fan, actions, z_true)
        onehot = torch.nn.functional.one_hot(actions[:, 0], count).float()
        packs["context_action"].append(
            torch.cat((root_features.repeat_interleave(count, 0), onehot), -1).cpu())
        packs["successor_cls"].append(cls_true[:, 0].cpu())
        packs["successor_z"].append(z_true[:, 0, 0].cpu())
        # The transition's own latent, on the same axis convention the ladder read z_true on.
        packs["generated_z"].append(advanced.latent[:, 0, 0].cpu())
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
    data["row_identity"] = identity.hexdigest()
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", type=Path, default=HERE / "evidence/root_partition.json")
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    parser.add_argument("--fit-roots", type=int, default=8000)
    parser.add_argument("--judge-roots", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    from d4mj.experiments import _load_bridge_parent
    from d4mj.lewm_config import window_layout

    checkpoint = args.checkpoint or args.run / "bridge/step-002000.pt"
    bundle, heads, payload = _load_bridge_parent(checkpoint)
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
    heads.eval()
    config = bundle.config
    offsets = window_layout(config.joint)[0]
    span = offsets[-1] + 1
    device, count = bundle.device, bundle.n_actions
    width, bins = config.dynamics.width, config.agent.bins
    centers = heads.centers.detach().clone().to(device)

    partition = json.loads(args.partition.read_text())
    paths = {int(q.stem.split("-")[1]): q for q in Path(FORK_STORE).glob("seed-*.pt")}
    fit_seeds, judge_seeds = seeds_for(partition, FORK_STORE)
    if set(fit_seeds) & set(judge_seeds):
        raise SystemExit("fit and judgement seeds overlap")
    from ladder import rows_for
    rows = {"fit": rows_for(fit_seeds, span, args.fit_roots),
            "judge": rows_for(judge_seeds, span, args.judge_roots)}
    splits = {}
    for name in ("fit", "judge"):
        splits[name] = materialize(bundle, rows[name], span)
        print(json.dumps({"stage": "materialized", "split": name, "roots": splits[name]["roots"],
                          "rows": int(len(splits[name]["reward"])),
                          "seconds": round(time.time() - started, 1)}), flush=True)

    # ---- 1. the TRAINED bridge head, on the judgement roots, read by the closure's own code ----
    judged = pack(rows["judge"], span)
    read = _fork_readouts(bundle, heads, judged)
    trained = {}
    for label, reward_key, death_key in (("real", "true_reward", "true_death"),
                                         ("generated", "generated_reward", "generated_death")):
        trained[label] = decisions(splits["judge"]["reward"], splits["judge"]["terminated"],
                                   read[reward_key].reshape(-1), read[death_key].reshape(-1),
                                   splits["judge"]["seed"], count, draws=args.draws,
                                   seed=args.seed + 7)
    print(json.dumps({"stage": "trained_head_scored",
                      "real_safe": trained["real"]["terminal"].get("safe_choice"),
                      "generated_safe": trained["generated"]["terminal"].get("safe_choice"),
                      "marginal_safe": trained["real"]["terminal"].get("marginal_safe_choice"),
                      "seconds": round(time.time() - started, 1)}), flush=True)

    # The trained head's per-root vectors, kept so every fresh head can be contrasted against it
    # DIRECTLY on these roots, rather than inferred from two intervals sharing a reference.
    trained_vectors = {key: value for key, value in trained["real"]["_vectors"].items()}
    trained_generated = {key: value for key, value in trained["generated"]["_vectors"].items()}

    del bundle, heads, payload, read, judged
    torch.cuda.empty_cache()

    # ---- 2-4. fresh heads: the exact deployed capacity, and the adapter control ----
    fit_roots = splits["fit"]["roots"]
    inner = torch.zeros(fit_roots, dtype=torch.bool)
    inner[torch.randperm(fit_roots, generator=torch.Generator().manual_seed(args.seed + 3))
          [: max(1, int(round(0.15 * fit_roots)))]] = True
    holdout = inner.repeat_interleave(count)

    control, matrix = {}, {name: {} for name in FAMILIES}
    for variant, make in (("adapter", Readout), ("exact", Exact)):
        for family in FAMILIES:
            if variant == "exact" and family not in EXACT_OK:
                continue
            model = fit(splits["fit"][family], splits["fit"]["reward"],
                        ~splits["fit"]["terminated"], centers, steps=args.steps, seed=args.seed,
                        device=device, width=width, bins=bins, holdout=holdout, make=make)
            for target in CROSS[family]:
                if variant == "exact" and target not in EXACT_OK:
                    continue
                cell = {"fitted_on": family, "evaluated_on": target, "variant": variant}
                for split in ("fit", "judge"):
                    data = splits[split]
                    pr, pd = predict(model, data[target], centers, device)
                    cell[split] = decisions(data["reward"], data["terminated"], pr, pd,
                                            data["seed"], count, draws=args.draws,
                                            seed=args.seed + 7)
                if variant == "adapter" and target == "context_action":
                    control = {(s, k): v for s in ("fit", "judge")
                               for k, v in cell[s]["_vectors"].items()}
                for split in ("fit", "judge"):
                    for key, higher in (("reward", False), ("terminal", True)):
                        mine = cell[split]["_vectors"].get(key)
                        if mine is None:
                            continue
                        against = [("vs_context_action", control.get((split, key)))]
                        if split == "judge":
                            # The trained head read the REAL successor for real-feature targets and
                            # the generated one for generated-feature targets; contrast like with like.
                            reference = (trained_vectors if target != "generated_features"
                                         else trained_generated).get(key)
                            against.append(("vs_trained_head", reference))
                        for name, theirs in against:
                            if theirs is None or (name == "vs_context_action"
                                                  and target == "context_action"):
                                continue
                            left, right = (mine[0], theirs[0]) if higher else (theirs[0], mine[0])
                            cell[split][key][name] = paired(
                                left, right, mine[1], draws=args.draws, seed=args.seed + 11)
                    del cell[split]["_vectors"]
                # `_gap` names the ladder's splits; this run calls them fit/judge.
                cell["train_dev_gap"] = _gap({"train": cell["fit"], "dev": cell["judge"]})
                cell["fit_trace"] = model.trace
                matrix[target][f"{variant}:{family}"] = cell
                print(json.dumps({"stage": "scored", "variant": variant, "fitted_on": family,
                                  "evaluated_on": target,
                                  "judge_reward_beats": cell["judge"]["reward"].get("beats_marginal"),
                                  "judge_terminal_beats": cell["judge"]["terminal"].get("beats_marginal"),
                                  "seconds": round(time.time() - started, 1)}), flush=True)
            del model
            torch.cuda.empty_cache()

    for label in trained:
        trained[label].pop("_vectors", None)

    report = {"schema": "d4mj_readout_confirm_v1",
              "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint),
              "partition_sha256": _sha256(args.partition), "script_sha256": _sha256(Path(__file__)),
              "ladder_sha256": _sha256(HERE / "ladder.py"),
              # Item 5: the fork store carries no manifest, so bind the bytes AND the exact rows.
              # Filenames are `seed-<padded>-r<roots>.pt`, so the path cannot be rebuilt from the
              # seed number; the map is built by globbing, exactly as `rows_for` reads them.
              "fork_store": str(FORK_STORE),
              "fork_files_sha256": hashlib.sha256("".join(
                  _sha256(paths[s]) for s in fit_seeds + judge_seeds
              ).encode()).hexdigest(),
              "row_identity": {name: splits[name]["row_identity"] for name in splits},
              "judge_seeds": {"count": len(judge_seeds),
                              "sha256": hashlib.sha256(repr(judge_seeds).encode()).hexdigest(),
                              "note": "the 405 seeds the sealed partition left unallocated; no head "
                                      "in this run was fitted or selected on any of them"},
              "roots": {name: splits[name]["roots"] for name in splits},
              "steps": args.steps, "draws": args.draws, "seed": args.seed,
              "matrix_layout": "matrix[evaluated_on]['<variant>:<fitted_on>'][split]",
              "trained_bridge_head": trained,
              "matrix": matrix}
    (args.out / "confirm.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "confirm_complete", "seconds": round(time.time() - started, 1)}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
