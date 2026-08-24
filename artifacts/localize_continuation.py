"""Localize where terminal information is lost in Direct-Attention.

Diagnostic only. It does not modify training.

For each genuine terminal transition, construct a matched alive transition from
another episode at the SAME absolute environment timestep. Both examples occupy
the SAME final sequence slot and are each reached by exactly ONE transition from
their own observed predecessor.

Probe, in order:
  1. real encoder latent z_t
  2. observed world agent readout h_t
  3. one-step generated latent zhat_t
  4. one-step generated world agent readout hhat_t
  5. production continuation head on observed/generated readouts

Controls:
  * action-only one-hot probe
  * absolute-position-only probe, matched by construction

Each linear/MLP probe uses a TRAIN-fit / TRAIN-validation split by whole episode,
selects the best checkpoint on TRAIN-validation BCE, reports untouched DEV, and
repeats across multiple seeds.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from artifacts.run_stage_a import corpus
from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import Batch, Episode, _window
from d4mj.state import WorldState
from d4mj.train import _to, cache_latents, train_representation
from d4mj.transition import World, advance, commit_inputs


@dataclass(frozen=True)
class TransitionRef:
    episode: int
    action_index: int


@dataclass(frozen=True)
class Pair:
    dead: TransitionRef
    alive: TransitionRef


def _auc(score: Tensor, target: Tensor) -> float:
    score = score.detach().float().cpu().flatten()
    target = target.detach().bool().cpu().flatten()
    pos, neg = score[target], score[~target]
    if not len(pos) or not len(neg):
        return 0.5
    d = pos[:, None] - neg[None]
    return float((d.gt(0).float() + 0.5 * d.eq(0).float()).mean())


def _binary_metrics(probability: Tensor, target: Tensor) -> dict[str, float]:
    p = probability.detach().float().cpu().flatten().clamp(1e-7, 1 - 1e-7)
    y = target.detach().float().cpu().flatten()
    dead = y.bool()
    return {
        "bce": float(F.binary_cross_entropy(p, y)),
        "auc": _auc(p, dead),
        "accuracy": float(((p >= 0.5) == dead).float().mean()),
        "mean_on_dead": float(p[dead].mean()) if bool(dead.any()) else float("nan"),
        "mean_on_alive": float(p[~dead].mean()) if bool((~dead).any()) else float("nan"),
    }


def _eligible_dead(episodes: list[Episode], length: int) -> list[TransitionRef]:
    out = []
    minimum_t = length - 2
    for ei, episode in enumerate(episodes):
        if not episode.uniform_eligible or episode.latents is None:
            continue
        for t in episode.terminated.nonzero().flatten().tolist():
            t = int(t)
            if t >= minimum_t:
                out.append(TransitionRef(ei, t))
    return out


def _alive_by_position(episodes: list[Episode], length: int) -> dict[int, list[TransitionRef]]:
    out: dict[int, list[TransitionRef]] = {}
    minimum_t = length - 2
    for ei, episode in enumerate(episodes):
        if not episode.uniform_eligible or episode.latents is None:
            continue
        for t in range(max(0, minimum_t), len(episode)):
            if bool(episode.terminated[t]) or bool(episode.truncated[t]):
                continue
            out.setdefault(t, []).append(TransitionRef(ei, t))
    return out


def make_pairs(episodes: list[Episode], length: int, *, seed: int) -> list[Pair]:
    """Match each death to an alive transition at the exact same env timestep."""
    dead = _eligible_dead(episodes, length)
    alive = _alive_by_position(episodes, length)
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(dead), generator=rng).tolist() if dead else []

    pairs = []
    usage: dict[tuple[int, int], int] = {}
    for di in order:
        d = dead[di]
        candidates = [a for a in alive.get(d.action_index, []) if a.episode != d.episode]
        if not candidates:
            continue
        counts = torch.tensor(
            [usage.get((a.episode, a.action_index), 0) for a in candidates],
            dtype=torch.long,
        )
        minimum = int(counts.min())
        tied = [a for a, c in zip(candidates, counts.tolist()) if c == minimum]
        a = tied[int(torch.randint(len(tied), (1,), generator=rng))]
        usage[(a.episode, a.action_index)] = minimum + 1
        pairs.append(Pair(d, a))

    if not pairs:
        raise RuntimeError("No position-matched alive/dead pairs could be constructed")
    return pairs


def _row_for(episode: Episode, action_index: int, length: int, config: Config) -> dict[str, Tensor]:
    # Final block is successor observation action_index + 1.
    start = action_index - length + 2
    assert start >= 0
    row = _window(episode, start, length, config)
    assert bool(row["valid"][-1])
    return row


def make_batch(
    episodes: list[Episode],
    pairs: list[Pair],
    length: int,
    config: Config,
) -> tuple[Batch, Tensor, Tensor, Tensor]:
    """Dead then alive for each pair; return death labels, final actions, positions."""
    rows = []
    target = []
    positions = []

    for pair in pairs:
        for ref, is_dead in ((pair.dead, True), (pair.alive, False)):
            row = _row_for(episodes[ref.episode], ref.action_index, length, config)
            actual_dead = bool(row["terminated"][-1])
            if actual_dead != is_dead:
                raise AssertionError(f"label mismatch: expected dead={is_dead}, got {actual_dead}")
            rows.append(row)
            target.append(float(is_dead))
            positions.append(float(ref.action_index))

    stack = {field: torch.stack([row[field] for row in rows]) for field in rows[0]}
    batch = Batch(burn_in=0, relevant=None, support=None, **stack)
    target_t = torch.tensor(target, dtype=torch.float32)
    action = batch.led_to_action[:, -1].long()
    position = torch.tensor(positions, dtype=torch.float32)

    if not torch.equal(position[0::2], position[1::2]):
        raise AssertionError("alive/dead examples are not exactly position matched")
    return batch, target_t, action, position


@torch.no_grad()
def extract(
    world: World,
    episodes: list[Episode],
    pairs: list[Pair],
    config: Config,
    *,
    length: int,
    seed: int,
    chunk_size: int,
) -> dict[str, Tensor]:
    """Extract four stages from exactly one final semantic transition."""
    real_latent = []
    observed_readout = []
    generated_latent = []
    generated_readout = []
    labels = []
    actions = []
    positions = []
    rng = torch.Generator(device=config.device).manual_seed(seed)

    for start in range(0, len(pairs), chunk_size):
        subset = pairs[start : start + chunk_size]
        batch_cpu, target, action, position = make_batch(episodes, subset, length, config)
        batch = _to(batch_cpu, config.device)

        z_real = batch.latents[:, -1:].detach()

        # Observed world readout after consuming the true successor.
        committed, conditioning = commit_inputs(batch.latents, rng, config)
        _, agent_real_full, _ = world(None, batch.led_to_action, committed, conditioning)
        h_real = agent_real_full[:, -1:].detach()

        # Build observed predecessor state from all blocks except the final one.
        prefix_committed, prefix_conditioning = commit_inputs(batch.latents[:, :-1], rng, config)
        prefix_features, _, prefix_memory = world(
            None,
            batch.led_to_action[:, :-1],
            prefix_committed,
            prefix_conditioning,
        )
        state = WorldState(
            batch.latents[:, -2:-1],
            prefix_memory,
            batch.latents.shape[1] - 1,
            prefix_features[:, -1:],
        )

        # One deployed Direct transition into the final slot.
        generated_state, h_generated = advance(
            world,
            state,
            batch.led_to_action[:, -1:],
            rng,
            config,
        )

        real_latent.append(z_real.cpu())
        observed_readout.append(h_real.cpu())
        generated_latent.append(generated_state.latent.detach().cpu())
        generated_readout.append(h_generated.detach().cpu())
        labels.append(target)
        actions.append(action)
        positions.append(position)

    return {
        "real_latent": torch.cat(real_latent),
        "observed_readout": torch.cat(observed_readout),
        "generated_latent": torch.cat(generated_latent),
        "generated_readout": torch.cat(generated_readout),
        "target": torch.cat(labels),
        "action": torch.cat(actions),
        "position": torch.cat(positions),
    }


def _flatten(x: Tensor) -> Tensor:
    return x.float().flatten(1)


def _standardize(fit_x: Tensor, *others: Tensor) -> tuple[Tensor, ...]:
    fit_x = fit_x.float()
    mean = fit_x.mean(0, keepdim=True)
    std = fit_x.std(0, unbiased=False, keepdim=True).clamp(min=1e-5)
    return (fit_x - mean) / std, *[(x.float() - mean) / std for x in others]


def _fit_linear_once(
    fit_x: Tensor,
    fit_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    evals: dict[str, tuple[Tensor, Tensor]],
    *,
    seed: int,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> dict:
    names = list(evals)
    standardized = _standardize(fit_x, val_x, *[evals[n][0] for n in names])
    fit_x, val_x, *extra = standardized
    eval_std = {name: (x, evals[name][1]) for name, x in zip(names, extra)}

    torch.manual_seed(seed)
    model = nn.Linear(fit_x.shape[1], 1).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sampler = torch.Generator().manual_seed(seed + 100_000)
    best_loss = float("inf")
    best_state = None

    for step in range(steps):
        idx = torch.randint(len(fit_x), (min(batch_size, len(fit_x)),), generator=sampler)
        x = fit_x[idx].to(device)
        y = fit_y[idx].to(device).float()
        logits = model(x)[:, 0]
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 10 == 0 or step == steps - 1:
            with torch.no_grad():
                val_logits = model(val_x.to(device))[:, 0]
                val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_y.to(device).float()))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    with torch.no_grad():
        for name, (x, y) in {
            "fit": (fit_x, fit_y),
            "validation": (val_x, val_y),
            **eval_std,
        }.items():
            p = model(x.to(device))[:, 0].sigmoid().cpu()
            out[name] = _binary_metrics(p, y)
    out["best_validation_bce"] = best_loss
    return out


def _summarize_runs(runs: list[dict]) -> dict:
    names = [k for k in runs[0] if isinstance(runs[0][k], dict)]
    out = {}
    for name in names:
        out[name] = {}
        for metric in runs[0][name]:
            values = torch.tensor([float(run[name][metric]) for run in runs], dtype=torch.float32)
            out[name][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(unbiased=False)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
    return out


def fit_linear(
    fit_x: Tensor,
    fit_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    evals: dict[str, tuple[Tensor, Tensor]],
    *,
    seeds: list[int],
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> dict:
    runs = [
        _fit_linear_once(
            fit_x, fit_y, val_x, val_y, evals,
            seed=seed,
            device=device,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        for seed in seeds
    ]
    return {"runs": runs, "summary": _summarize_runs(runs)}


def _continuation_logits(heads: Heads, agent: Tensor) -> Tensor:
    pooled = agent.mean(dim=2)
    model = heads.model_body(pooled)
    return heads.continuation(model)[..., 0]


def _fit_mlp_once(
    fit_agent: Tensor,
    fit_y: Tensor,
    val_agent: Tensor,
    val_y: Tensor,
    evals: dict[str, tuple[Tensor, Tensor]],
    config: Config,
    *,
    seed: int,
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> dict:
    torch.manual_seed(seed)
    heads = Heads(config).to(config.device)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.model_body.parameters():
        parameter.requires_grad_(True)
    for parameter in heads.continuation.parameters():
        parameter.requires_grad_(True)

    params = [p for p in heads.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sampler = torch.Generator().manual_seed(seed + 200_000)
    best_loss = float("inf")
    best_body = None
    best_head = None

    for step in range(steps):
        idx = torch.randint(len(fit_agent), (min(batch_size, len(fit_agent)),), generator=sampler)
        agent = fit_agent[idx].to(config.device)
        y = fit_y[idx].to(config.device).float()
        # Head predicts continuation, target here is death.
        logits = _continuation_logits(heads, agent).flatten()
        loss = F.binary_cross_entropy_with_logits(logits, 1.0 - y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 10 == 0 or step == steps - 1:
            heads.eval()
            with torch.no_grad():
                logits = _continuation_logits(heads, val_agent.to(config.device)).flatten()
                death_p = 1.0 - logits.sigmoid()
                val_loss = float(F.binary_cross_entropy(
                    death_p.clamp(1e-7, 1 - 1e-7), val_y.to(config.device).float()
                ))
            heads.train()
            if val_loss < best_loss:
                best_loss = val_loss
                best_body = copy.deepcopy(heads.model_body.state_dict())
                best_head = copy.deepcopy(heads.continuation.state_dict())

    assert best_body is not None and best_head is not None
    heads.model_body.load_state_dict(best_body)
    heads.continuation.load_state_dict(best_head)
    heads.eval()
    out = {}
    with torch.no_grad():
        for name, (agent, y) in {
            "fit": (fit_agent, fit_y),
            "validation": (val_agent, val_y),
            **evals,
        }.items():
            logits = _continuation_logits(heads, agent.to(config.device)).flatten()
            death_p = (1.0 - logits.sigmoid()).cpu()
            out[name] = _binary_metrics(death_p, y)
    out["best_validation_bce"] = best_loss
    return out


def fit_mlp(
    fit_agent: Tensor,
    fit_y: Tensor,
    val_agent: Tensor,
    val_y: Tensor,
    evals: dict[str, tuple[Tensor, Tensor]],
    config: Config,
    *,
    seeds: list[int],
    steps: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
) -> dict:
    runs = [
        _fit_mlp_once(
            fit_agent, fit_y, val_agent, val_y, evals, config,
            seed=seed,
            steps=steps,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )
        for seed in seeds
    ]
    return {"runs": runs, "summary": _summarize_runs(runs)}


@torch.no_grad()
def production_head_metrics(heads: Heads, data: dict[str, Tensor], config: Config) -> dict[str, dict[str, float]]:
    heads.eval()
    out = {}
    for name in ("observed_readout", "generated_readout"):
        logits = _continuation_logits(heads, data[name].to(config.device)).flatten()
        death_p = (1.0 - logits.sigmoid()).cpu()
        out[name] = _binary_metrics(death_p, data["target"])
    return out


def one_hot_actions(data: dict[str, Tensor], n_actions: int) -> Tensor:
    return F.one_hot(data["action"], num_classes=n_actions).float()


def position_features(data: dict[str, Tensor]) -> Tensor:
    p = data["position"].float()
    return torch.stack([p, torch.log1p(p)], dim=1)


def split_train_episodes(episodes: list[Episode], *, seed: int, fraction: float) -> tuple[list[Episode], list[Episode]]:
    order = torch.randperm(len(episodes), generator=torch.Generator().manual_seed(seed))
    cut = max(1, min(len(episodes) - 1, int(len(episodes) * fraction)))
    return (
        [episodes[int(i)] for i in order[:cut]],
        [episodes[int(i)] for i in order[cut:]],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", type=Path, required=True, help="Stage-A directory containing phase1a.pt")
    parser.add_argument("--phase2", type=Path, required=True, help="Direct-Attention Phase-2 checkpoint")
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--tokenizer-steps", type=int, default=3000)
    parser.add_argument("--length", type=int, default=16)
    parser.add_argument("--fit-fraction", type=float, default=0.8)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-lr", type=float, default=1e-2)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-3)
    parser.add_argument("--mlp-steps", type=int, default=2000)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-batch", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("artifacts/continuation_localization.json"))
    args = parser.parse_args()

    if args.length < 2:
        parser.error("--length must be at least 2")

    base = Config()
    config = replace(base, transition="direct", time_mixer="attention")

    train_set, dev_set = corpus(base, args.expert, print)
    fit_raw, val_raw = split_train_episodes(
        train_set, seed=config.seed + 1200, fraction=args.fit_fraction
    )

    # Restore tokenizer and rebuild frozen latent caches.
    encoder, _, fit_cached = train_representation(
        fit_raw,
        args.tokenizer_steps,
        base,
        checkpoint=args.reuse / "phase1a.pt",
    )
    val_cached = cache_latents(encoder, val_raw, base)
    dev_cached = cache_latents(encoder, dev_set, base)
    encoder.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load the actual Phase-2 world and continuation head. train_agent checkpoints
    # store world as part0 and heads as part1.
    world = World(config).to(config.device)
    heads = Heads(config).to(config.device)
    load(args.phase2, config, part0=world, part1=heads)
    world.eval()
    heads.eval()
    for module in (world, heads):
        for parameter in module.parameters():
            parameter.requires_grad_(False)

    fit_pairs = make_pairs(fit_cached, args.length, seed=config.seed + 1300)
    val_pairs = make_pairs(val_cached, args.length, seed=config.seed + 1400)
    dev_pairs = make_pairs(dev_cached, args.length, seed=config.seed + 1500)
    print(f"pairs: fit={len(fit_pairs)} validation={len(val_pairs)} dev={len(dev_pairs)}", flush=True)

    fit = extract(
        world, fit_cached, fit_pairs, config,
        length=args.length, seed=config.seed + 1600, chunk_size=args.feature_batch,
    )
    val = extract(
        world, val_cached, val_pairs, config,
        length=args.length, seed=config.seed + 1700, chunk_size=args.feature_batch,
    )
    dev = extract(
        world, dev_cached, dev_pairs, config,
        length=args.length, seed=config.seed + 1800, chunk_size=args.feature_batch,
    )

    seeds = [config.seed + 2000 + i for i in range(args.seeds)]
    linear = {}

    for name in ("real_latent", "observed_readout", "generated_latent", "generated_readout"):
        print(f"linear probe: {name}", flush=True)
        evals = {"dev_same_path": (_flatten(dev[name]), dev["target"])}
        if name == "real_latent":
            evals["dev_cross_generated"] = (_flatten(dev["generated_latent"]), dev["target"])
        elif name == "generated_latent":
            evals["dev_cross_real"] = (_flatten(dev["real_latent"]), dev["target"])
        elif name == "observed_readout":
            evals["dev_cross_generated"] = (_flatten(dev["generated_readout"]), dev["target"])
        else:
            evals["dev_cross_observed"] = (_flatten(dev["observed_readout"]), dev["target"])

        linear[name] = fit_linear(
            _flatten(fit[name]), fit["target"],
            _flatten(val[name]), val["target"],
            evals,
            seeds=seeds,
            device=config.device,
            steps=args.probe_steps,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            batch_size=args.batch_size,
        )

    print("linear probe: action_only", flush=True)
    linear["action_only"] = fit_linear(
        one_hot_actions(fit, config.n_actions), fit["target"],
        one_hot_actions(val, config.n_actions), val["target"],
        {"dev_same_path": (one_hot_actions(dev, config.n_actions), dev["target"])},
        seeds=seeds,
        device=config.device,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.batch_size,
    )

    print("linear probe: position_only", flush=True)
    linear["position_only"] = fit_linear(
        position_features(fit), fit["target"],
        position_features(val), val["target"],
        {"dev_same_path": (position_features(dev), dev["target"])},
        seeds=seeds,
        device=config.device,
        steps=args.probe_steps,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.batch_size,
    )

    print("production-shaped MLP probe: observed_readout", flush=True)
    mlp_observed = fit_mlp(
        fit["observed_readout"], fit["target"],
        val["observed_readout"], val["target"],
        {
            "dev_same_path": (dev["observed_readout"], dev["target"]),
            "dev_cross_generated": (dev["generated_readout"], dev["target"]),
        },
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.batch_size,
    )

    print("production-shaped MLP probe: generated_readout", flush=True)
    mlp_generated = fit_mlp(
        fit["generated_readout"], fit["target"],
        val["generated_readout"], val["target"],
        {
            "dev_same_path": (dev["generated_readout"], dev["target"]),
            "dev_cross_observed": (dev["observed_readout"], dev["target"]),
        },
        config,
        seeds=seeds,
        steps=args.mlp_steps,
        lr=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        batch_size=args.batch_size,
    )

    report = {
        "contract": {
            "arm": "direct-attention",
            "same_final_sequence_slot": True,
            "one_transition_from_own_predecessor": True,
            "absolute_environment_timestep_matched": True,
            "train_validation_split_by_whole_episode": True,
            "dev_untouched": True,
            "linear_probe_seeds": args.seeds,
            "same_pre_action_counterfactual_pairs": False,
            "same_pre_action_counterfactual_note": (
                "The d4mj Episode replay retains only the realized trajectory, not "
                "reconstructable simulator state for alternative actions."
            ),
        },
        "pairs": {"fit": len(fit_pairs), "validation": len(val_pairs), "dev": len(dev_pairs)},
        "production_head": {
            "fit": production_head_metrics(heads, fit, config),
            "validation": production_head_metrics(heads, val, config),
            "dev": production_head_metrics(heads, dev, config),
        },
        "linear_probes": linear,
        "production_shaped_mlp": {
            "observed_fit": mlp_observed,
            "generated_fit": mlp_generated,
        },
        "latent_generation_error": {
            split: {
                "all_mse": float((data["generated_latent"] - data["real_latent"]).pow(2).mean()),
                "dead_mse": float((
                    data["generated_latent"][data["target"].bool()]
                    - data["real_latent"][data["target"].bool()]
                ).pow(2).mean()),
                "alive_mse": float((
                    data["generated_latent"][~data["target"].bool()]
                    - data["real_latent"][~data["target"].bool()]
                ).pow(2).mean()),
            }
            for split, data in (("fit", fit), ("validation", val), ("dev", dev))
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()