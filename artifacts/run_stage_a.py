"""Stage-A preliminary run: one shared tokenizer, then four arms through 1B, 2, 3.

Phase 1A is trained *once*. The tokenizer is always attention (S20) and the cache
digest deliberately excludes the time mixer, so one encoder and one latent cache
serve all four arms -- which is also what makes the comparison a comparison.

This is a DEV run. Evaluation here is a shortened smoke, not the preregistered FINAL
protocol (S52): the sealed FINAL seeds are untouched and the native 10000-step
horizon is not used, because 32 episodes at that horizon costs hours per arm.
Nothing here may be reported as a Stage-A result.
"""

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from d4mj.config import Config
from d4mj.counterfactual import (
    actor_safety_metrics,
    collect_outcome_forks,
    outcome_metrics,
)
from d4mj.data import EpisodeCorpus, episode_splits, load_episodes
from d4mj.diagnostics import cost, head_calibration, latent_stats, multistep_error
from d4mj.execution import evaluate, run_episode, run_random
from d4mj.expert import load_archive
from d4mj.train import (
    cache_latents,
    train_actor,
    train_agent,
    train_dynamics,
    train_representation,
)
from d4mj.transition import transition_loss

ARCHIVE = Path("d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt")
SUPPORT = Path("artifacts/craftax_support_v1.pt")
ARMS = tuple(
    f"{transition}-{mixer}"
    for transition in ("flow", "direct")
    for mixer in ("attention", "mamba")
)


def corpus(
    config: Config,
    expert: int,
    log,
    *,
    support: Path = SUPPORT,
) -> tuple[EpisodeCorpus, EpisodeCorpus]:
    """Split each pool separately. A single split over the concatenation can hand DEV
    no BC-eligible episodes at all, and the mixture then cannot form its relevant
    half -- which is a property of the split, not of the data."""
    pools = [load_archive(ARCHIVE, config, limit=expert)]
    if support.exists():
        pools.append(load_episodes(support))

    train, dev = [], []
    for index, pool in enumerate(pools):
        declared = [episode.split for episode in pool]
        if any(value is not None for value in declared):
            if any(value is None for value in declared):
                raise ValueError("an episode store mixes declared and undeclared splits")
            train += [episode for episode in pool if episode.split == "train"]
            dev += [episode for episode in pool if episode.split == "dev"]
        else:
            first, second, _ = episode_splits(len(pool), config.seed + index)
            train += [pool[i] for i in first.tolist()]
            dev += [pool[i] for i in second.tolist()]

    for name, part in (("train", train), ("dev", dev)):
        log(
            f"{name}: {len(part)} episodes, {sum(len(e) for e in part)} transitions, "
            f"{sum(int(e.terminated.sum()) for e in part)} terminals, "
            f"{sum(e.bc_eligible for e in part)} BC-eligible"
        )
    return EpisodeCorpus(train), EpisodeCorpus(dev)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--tokenizer-steps", type=int, default=3000)
    parser.add_argument("--dynamics-steps", type=int, default=20000)
    parser.add_argument("--agent-steps", type=int, default=10000)
    parser.add_argument("--actor-steps", type=int, default=2500)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--eval-limit", type=int, default=600)
    parser.add_argument(
        "--terminal-dynamics-mass",
        type=float,
        default=0.0,
        help="Phase-2 dynamics mass assigned to tail-aligned terminal rows",
    )
    parser.add_argument(
        "--phase2-only",
        action="store_true",
        help="stop each arm after Phase-2 diagnostics and the outcome gate",
    )
    parser.add_argument("--out", default="artifacts/stage_a")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument(
        "--reuse",
        type=Path,
        help="verified Phase-1A and Phase-1B checkpoints to reuse",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - started:7.0f}s] {message}"
        print(line, flush=True)
        (out / "run.log").open("a").write(line + "\n")

    base = Config()
    if not 0.0 <= args.terminal_dynamics_mass < 1.0:
        parser.error("--terminal-dynamics-mass must be in [0, 1)")
    if args.dynamics_steps <= base.bootstrap_start:
        parser.error(
            f"--dynamics-steps must exceed bootstrap_start={base.bootstrap_start}; "
            "otherwise shortcut self-training never occurs in Phase 1B"
        )
    log(
        f"terminal supervision: batch {base.terminal_batch} every Phase-2 step, "
        f"{base.terminal_loss_mass:.1%} of continuation loss; "
        f"{args.terminal_dynamics_mass:.1%} of dynamics loss"
    )
    log(f"arms: {args.arms}")
    if args.reuse:
        log(f"reusing Phase-1 checkpoints from {args.reuse.resolve()}")
    train_set, dev_set = corpus(base, args.expert, log)

    log(f"phase 1A: {args.tokenizer_steps} steps, shared by every arm")
    phase1a = args.reuse / "phase1a.pt" if args.reuse else out / "phase1a.pt"
    encoder, decoder, cached_train = train_representation(
        train_set, args.tokenizer_steps, base, checkpoint=phase1a
    )
    cached_dev = cache_latents(encoder, dev_set, base)
    log(f"phase 1A done, cache digest {cached_train[0].latent_digest}")
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, out / "tokenizer.pt")

    report: dict[str, dict] = {}
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        log(f"resuming, {sorted(report)} already complete")

    for transition in ("flow", "direct"):
        for mixer in ("attention", "mamba"):
            arm = f"{transition}-{mixer}"
            if arm not in args.arms:
                continue
            if arm in report:
                continue
            config = replace(base, transition=transition, time_mixer=mixer)
            continuation_objective = (
                "paired-observed-generated-alive-dead-v2"
                if transition == "direct"
                else "paired-single-readout-alive-dead-v2"
            )
            log(f"=== {arm} ===")
            log(f"{arm}: continuation objective {continuation_objective}")
            # Mamba's Triton autotuner benchmarks several kernel configs and needs
            # headroom the resident tokenizer was holding; it is unused until eval.
            encoder.cpu()
            torch.cuda.empty_cache()

            phase1b = args.reuse / f"{arm}.1b.pt" if args.reuse else out / f"{arm}.1b.pt"
            world = train_dynamics(cached_train, args.dynamics_steps, config, checkpoint=phase1b)
            log(f"{arm}: phase 1B done")
            heads = train_agent(
                cached_train,
                world,
                args.agent_steps,
                config,
                checkpoint=out / f"{arm}.2.pt",
                world_steps=args.dynamics_steps,
                terminal_dynamics_mass=args.terminal_dynamics_mass,
            )
            log(f"{arm}: phase 2 done")
            prior = copy.deepcopy(heads).eval()
            horizon, horizon_report = select_horizon(world, cached_dev, config, log)
            config = replace(config, horizon=horizon)
            forks = collect_outcome_forks(world, encoder, prior, config)
            torch.save(vars(forks), out / f"{arm}.outcome_forks.pt")
            outcome_gate = outcome_metrics(forks, prior, config)
            log(
                f"{arm}: outcome gate pass={outcome_gate['passed']} | "
                f"reward regret {outcome_gate['reward_choice_regret']:.4f} "
                f"vs {outcome_gate['reward_marginal_regret']:.4f} marginal | "
                f"terminal BCE {outcome_gate['terminal_bce']:.4f} "
                f"vs {outcome_gate['terminal_marginal_bce']:.4f} marginal | "
                f"AUC {outcome_gate['terminal_auc']:.3f}"
            )
            log(
                f"{arm}: observed-successor reward regret "
                f"{outcome_gate['observed_reward_choice_regret']:.4f} | "
                f"terminal BCE {outcome_gate['observed_terminal_bce']:.4f} | "
                f"AUC {outcome_gate['observed_terminal_auc']:.3f}"
            )
            if args.phase2_only or not outcome_gate["passed"]:
                report[arm] = {
                    "status": (
                        "phase2_only"
                        if args.phase2_only
                        else "outcome_gate_failed"
                    ),
                    "terminal_dynamics_mass": args.terminal_dynamics_mass,
                    "continuation_objective": continuation_objective,
                    "terminal_loss_mass": config.terminal_loss_mass,
                    "outcome_gate": outcome_gate,
                    "horizon_selection": horizon_report,
                    "diagnostics": _diagnostics(world, prior, cached_dev, config),
                }
                (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
                continue

            heads = train_actor(
                cached_train, world, heads, args.actor_steps, config, checkpoint=out / f"{arm}.3.pt"
            )
            log(f"{arm}: phase 3 done")
            actor_outcomes = outcome_metrics(forks, heads, config)
            actor_gate = actor_safety_metrics(outcome_gate, actor_outcomes)
            log(
                f"{arm}: actor gate pass={actor_gate['passed']} | "
                f"fork death change {actor_gate['true_death_change']:+.4f} | "
                f"immediate reward change {actor_gate['true_reward_change']:+.4f}"
            )

            entry = {
                "status": "complete" if actor_gate["passed"] else "actor_gate_failed",
                "continuation_objective": continuation_objective,
                "terminal_loss_mass": config.terminal_loss_mass,
                "terminal_dynamics_mass": args.terminal_dynamics_mass,
                "outcome_gate": outcome_gate,
                "actor_outcomes": actor_outcomes,
                "actor_gate": actor_gate,
                "cost": cost({"encoder": encoder, "world": world, "heads": heads}, world, config),
            }
            entry["diagnostics"] = _diagnostics(world, heads, cached_dev, config)
            entry["horizon_selection"] = horizon_report
            encoder.to(config.device)
            # The BC prior is the control S52 actually requires: beating a random
            # policy is not evidence, beating the behaviour the actor was cloned
            # from is the claim.
            scores = evaluate(
                {
                    "actor": lambda s: run_episode(
                        world, encoder, heads, s, config, limit=args.eval_limit
                    ),
                    "bc": lambda s: run_episode(
                        world, encoder, prior, s, config, limit=args.eval_limit
                    ),
                    "random": lambda s: run_random(s, config, limit=args.eval_limit),
                },
                list(range(10_000, 10_000 + args.eval_episodes)),
                config,
            )
            # S52 requires the raw rows kept: the official score is nonlinear, so a
            # paired interval cannot be reconstructed from summaries alone.
            entry["evaluation"] = {
                name: {k: v for k, v in row.items() if k != "episodes"} for name, row in scores.items()
            }
            entry["episodes"] = {
                name: [vars(r) for r in row["episodes"]] for name, row in scores.items()
            }
            report[arm] = entry
            log(
                f"{arm}: actor {scores['actor']['score']:.2f} bc {scores['bc']['score']:.2f} "
                f"random {scores['random']['score']:.2f} | "
                f"beats bc={scores['actor']['versus_bc']['beats']} "
                f"random={scores['actor']['versus_random']['beats']} | "
                f"continuation separation {entry['diagnostics']['continuation_separation']:.4f} | "
                f"contraction {entry['diagnostics']['contraction']:.3f} | horizon {config.horizon}"
            )
            (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    log("done")


def select_horizon(world, dev, config: Config, log) -> tuple[int, dict]:
    """S54, under the S63 criterion: the largest declared candidate at which the
    rollout still beats the marginal predictor -- the constant mean latent.

    That line is not a tuned threshold. Past it the rollout carries less about the
    future than knowing nothing does, so imagining further cannot inform the actor.
    It is also scale-free, which the previous rule was not: a tolerance relative to
    the one-step error is *tighter* for a more accurate arm, and it degenerated to
    "no candidate qualified" on three arms of four.

    `growth` is descriptive only: it is the per-step ratio of rolled MSE, which
    measures how error *accumulates*, not the perturbation sensitivity Lemma 1's
    hypothesis is about. The two are not the same -- an error recursion
    `e -> 0.5 e + 1` has sensitivity 0.5 yet an MSE ratio above 1 at every horizon --
    so this statistic cannot say whether that bound holds, and it is reported
    without that claim. Direct's horizon is additionally capped at the rollout
    length its loss actually trains (S68).
    """
    from d4mj.data import sample_batch
    from d4mj.train import _to

    reach = max(config.horizon_candidates)
    context = config.sequence_long - reach
    sampler = torch.Generator().manual_seed(config.seed + 555)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 556)

    rolled, marginal = [], []
    for repeat in range(8):
        batch = _to(sample_batch(dev, sampler, config, 4 * repeat + 3, 0, mixture=True), config.device)
        rolled.append(multistep_error(world, batch, rng, config, context=context)["mean_error"])
        future = batch.latents[:, context:]
        mean = batch.latents.mean(dim=(0, 1), keepdim=True)
        marginal.append([float((future[:, t] - mean[:, 0]).pow(2).mean()) for t in range(future.shape[1])])

    error = [sum(c[i] for c in rolled) / len(rolled) for i in range(len(rolled[0]))]
    trivial = [sum(c[i] for c in marginal) / len(marginal) for i in range(len(marginal[0]))]

    informative = [c for c in config.horizon_candidates if c <= len(error) and error[c - 1] < trivial[c - 1]]
    chosen = max(informative) if informative else min(config.horizon_candidates)
    capped = min(chosen, config.direct_rollout) if config.transition == "direct" else chosen
    growth = (error[chosen - 1] / error[0]) ** (1 / max(chosen - 1, 1))
    report = {
        "error": error,
        "marginal": trivial,
        "informative_horizon": chosen,
        "chosen": capped,
        "informative": bool(informative),
        "accumulation": growth,
    }
    log(
        f"horizon: rolled {[round(error[c - 1], 4) for c in config.horizon_candidates if c <= len(error)]} "
        f"vs marginal {[round(trivial[c - 1], 4) for c in config.horizon_candidates if c <= len(error)]} "
        f"-> informative {chosen}{'' if informative else ' (NONE, fell back)'}, "
        f"used {capped}{' (capped at trained rollout)' if capped != chosen else ''}; "
        f"error accumulation {growth:.4f}/step"
    )
    return capped, report


def _diagnostics(world, heads, dev, config: Config, batches: int = 200) -> dict:
    """Averaged over several DEV batches, because one batch carries too few terminals
    to say anything about continuation: the terminal-conditional probability is
    pooled by terminal count, not averaged over batches."""
    from d4mj.data import sample_batch, sample_terminal_batch
    from d4mj.train import _to
    from d4mj.transition import commit_inputs

    sampler = torch.Generator().manual_seed(config.seed + 999)
    rng = torch.Generator(device=config.device).manual_seed(config.seed + 998)
    totals: dict[str, float] = {}
    dead_mass = dead_count = 0.0

    for step in range(batches):
        batch = _to(sample_batch(dev, sampler, config, step, 0, mixture=True), config.device)
        with torch.no_grad():
            _, agent = transition_loss(world, batch, rng, config, return_agent=True)
        row = dict(latent_stats(world, batch, rng, config)) | head_calibration(
            heads, agent, batch, config
        )
        if row["terminal_targets"]:
            dead_mass += row["continuation_on_terminal"] * row["terminal_targets"]
            dead_count += row["terminal_targets"]
        for key, value in row.items():
            if key not in ("continuation_on_terminal", "continuation_separation"):
                totals[key] = totals.get(key, 0.0) + float(value)

    for step in range(32):
        batch = _to(sample_terminal_batch(dev, sampler, config, step, 32), config.device)
        with torch.no_grad():
            committed, conditioning = commit_inputs(batch.latents, rng, config)
            _, agent, _ = world(None, batch.led_to_action, committed, conditioning)
        row = head_calibration(heads, agent, batch, config)
        dead_mass += row["continuation_on_terminal"] * row["terminal_targets"]
        dead_count += row["terminal_targets"]

    out = {key: value / batches for key, value in totals.items()}
    out["terminal_targets"] = dead_count
    out["continuation_on_terminal"] = dead_mass / dead_count if dead_count else float("nan")
    out["continuation_separation"] = (
        out["continuation_on_continuing"] - out["continuation_on_terminal"]
        if dead_count
        else float("nan")
    )
    reach = max(config.horizon_candidates)
    batch = _to(sample_batch(dev, sampler, config, 3, 0, mixture=True), config.device)
    out["multistep"] = multistep_error(
        world, batch, rng, config, context=config.sequence_long - reach
    )["mean_error"]
    out["horizon"] = config.horizon
    return out


if __name__ == "__main__":
    main()
