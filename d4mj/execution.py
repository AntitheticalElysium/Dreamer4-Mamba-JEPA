import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .agent import Heads
from .config import Config
from .env import reset, step
from .representation import Encoder
from .transition import World
from .world_api import ModelBundle


@dataclass(frozen=True)
class Result:
    """One executed episode, kept raw. The official score is nonlinear in the
    per-achievement rates, so it cannot be averaged from per-episode scores and
    every interval has to be recomputed from these rows."""

    seed: int
    steps: int
    reward: float
    terminated: bool
    truncated: bool
    achievements: tuple[bool, ...]

    @property
    def unlocked(self) -> int:
        return sum(self.achievements)


def run_episode(
    world: World,
    encoder: Encoder,
    heads: Heads,
    seed: int,
    config: Config,
    limit: int | None = None,
    greedy: bool = False,
) -> Result:
    """The deployed loop, and the only place the whole system runs together.

    The horizon defaults to Craftax's native 10000, not the collector's 2500 cap
    (S50). Sampling is categorical at temperature 1, greedy only as a declared
    secondary (S52). Policy, flow corruption and environment draw from three
    separately seeded streams, so the arms do not differ by their own randomness.
    """
    bundle = ModelBundle.from_models(config, encoder, world)
    bundle.require_control()
    device = config.device
    bundle.world.to(device)
    bundle.encoder.to(device)
    bundle.eval()
    heads = heads.to(device).eval()
    rng = torch.Generator(device=device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=device).manual_seed(seed + 2**20)
    observation, env_state = reset(seed)
    state, total = None, 0.0
    action = None
    horizon = config.horizon_eval if limit is None else limit

    with torch.no_grad():
        for index in range(horizon):
            state, agent = bundle.observe(state, action, observation[None, None], rng)
            logits = heads(agent)["policy"][:, -1, 0]
            choice = (
                int(logits.argmax(-1))
                if greedy
                else int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            )
            observation, env_state, reward, terminated, truncated = step(
                env_state, choice, seed + index + 1
            )
            total += reward
            action = torch.full((1, 1), choice, dtype=torch.long, device=device)
            if terminated or truncated:
                return _result(seed, index + 1, total, terminated, truncated, env_state)
    return _result(seed, horizon, total, False, True, env_state)


def run_random(seed: int, config: Config, limit: int | None = None) -> Result:
    """The random control, on the same seed schedule as every other policy."""
    policy_rng = torch.Generator().manual_seed(seed + 2**20)
    observation, env_state = reset(seed)
    total = 0.0
    for index in range(config.horizon_eval if limit is None else limit):
        choice = int(torch.randint(config.n_actions, (1,), generator=policy_rng))
        observation, env_state, reward, terminated, truncated = step(
            env_state, choice, seed + index + 1
        )
        total += reward
        if terminated or truncated:
            return _result(seed, index + 1, total, terminated, truncated, env_state)
    return _result(seed, config.horizon_eval if limit is None else limit, total, False, True, env_state)


def score(results: list[Result]) -> float:
    """Craftax's official score: the geometric mean of per-achievement success
    rates in percent, `exp(mean(log(1 + rate))) - 1`.

    Computed over the whole set, never averaged from per-episode scores -- the log
    makes it nonlinear, so a mean of episode scores is a different statistic that
    merely looks similar. The `1 +` keeps one unattempted achievement from sending
    the score to zero.
    """
    if not results:
        return 0.0
    rates = torch.tensor([list(r.achievements) for r in results], dtype=torch.float64).mean(0) * 100
    return float(torch.expm1(torch.log1p(rates).mean()))


def evaluate(
    policies: dict[str, Callable[[int], Result]], seeds: list[int], config: Config
) -> dict[str, dict]:
    """Execute policies on paired seeds and bootstrap all reported metrics."""
    rows = {name: [policy(seed) for seed in seeds] for name, policy in policies.items()}
    generator = torch.Generator().manual_seed(config.seed + 2**22)
    draws = torch.randint(len(seeds), (config.bootstrap, len(seeds)), generator=generator).tolist()

    report: dict[str, dict] = {}
    for name, results in rows.items():
        samples = torch.tensor([score([results[i] for i in draw]) for draw in draws])
        reward_samples = torch.tensor([
            sum(results[i].reward for i in draw) / len(draw) for draw in draws
        ])
        achievement_samples = torch.tensor([
            sum(results[i].unlocked for i in draw) / len(draw) for draw in draws
        ])
        report[name] = {
            "score": score(results),
            "score_interval": _interval(samples),
            "reward": sum(r.reward for r in results) / len(results),
            "reward_interval": _interval(reward_samples),
            "achievements": sum(r.unlocked for r in results) / len(results),
            "achievements_interval": _interval(achievement_samples),
            "rates": [float(sum(r.achievements[i] for r in results)) / len(results)
                      for i in range(len(results[0].achievements))],
            "terminated": sum(r.terminated for r in results) / len(results),
            "length": sum(r.steps for r in results) / len(results),
            "episodes": results,
        }
    for name in rows:
        for control in rows:
            if name != control:
                gaps = torch.tensor([
                    score([rows[name][i] for i in draw]) - score([rows[control][i] for i in draw])
                    for draw in draws
                ])
                low, high = _interval(gaps)
                reward_gaps = torch.tensor([
                    sum(rows[name][i].reward - rows[control][i].reward for i in draw)
                    / len(draw)
                    for draw in draws
                ])
                reward_low, reward_high = _interval(reward_gaps)
                achievement_gaps = torch.tensor([
                    sum(rows[name][i].unlocked - rows[control][i].unlocked for i in draw)
                    / len(draw)
                    for draw in draws
                ])
                achievement_low, achievement_high = _interval(achievement_gaps)
                report[name][f"versus_{control}"] = {
                    "gap": report[name]["score"] - report[control]["score"],
                    "interval": (low, high),
                    "beats": low > 0.0,
                    "reward_gap": report[name]["reward"] - report[control]["reward"],
                    "reward_interval": (reward_low, reward_high),
                    "reward_beats": reward_low > 0.0,
                    "achievements_gap": (
                        report[name]["achievements"] - report[control]["achievements"]
                    ),
                    "achievements_interval": (achievement_low, achievement_high),
                    "achievements_beats": achievement_low > 0.0,
                }
    return report


def evaluate_lewm_actor(actor_path, output, *, episodes: int | None = None,
                        seed_base: int = 30_000, limit: int | None = None) -> dict:
    """Execute a self-contained canonical actor against its immutable own-BC prior.

    Episode cache identity includes every model byte and protocol choice and is atomically
    replaced, so interruption loses at most the current 16-episode block and never corrupts the
    reusable prefix.
    """
    from dataclasses import asdict
    from pathlib import Path
    import tempfile
    from .checkpoint import read_lewm_actor
    from .config import config_from_dict
    from .data import _sha256, atomic_manifest

    actor_path, output = Path(actor_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    payload = read_lewm_actor(actor_path)
    config = config_from_dict(payload["config"])
    settings = config.agent
    if not payload["capabilities"].get("actor_trained"):
        raise ValueError("phase_gate: primary execution requires the completed actor budget")
    if config.runtime.purpose == "research" and (
        episodes not in (None, settings.eval_episodes) or limit not in (None, settings.horizon_eval)
    ):
        raise ValueError("evaluation_protocol: research runs use the sealed episode count and native cap")
    count = settings.eval_episodes if episodes is None else episodes
    horizon = settings.horizon_eval if limit is None else limit
    if count < 1 or horizon < 1:
        raise ValueError("evaluation_protocol: episodes and horizon must be positive")

    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"], strict=True)
    bundle.world.load_state_dict(payload["modules"]["world"], strict=True)
    bundle.capabilities = dict(payload["capabilities"])
    bundle.eval().world.requires_grad_(False)
    bundle.encoder.freeze()
    policies = {}
    for name, key in (("actor", "heads"), ("bc", "prior")):
        head = Heads(config).to(config.runtime.device)
        head.load_state_dict(payload["modules"][key], strict=True)
        policies[name] = head.eval().requires_grad_(False)
    seeds = list(range(seed_base, seed_base + count))
    identity = {
        "schema": "d4mj_lewm_execution_identity_v1", "actor_sha256": _sha256(actor_path),
        "recipe_id": payload["recipe_id"], "seed_base": seed_base, "episodes": count,
        "limit": horizon, "protocol": "categorical-temperature-1-own-bc-paired-v1",
    }
    cache_path = output / "episodes.pt"
    cached_payload = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path.exists() else {}
    if cached_payload and cached_payload.get("identity") != identity:
        raise ValueError("execution_cache: output contains a different actor/protocol")
    rows = cached_payload.get("rows", {})

    def save_cache():
        with tempfile.NamedTemporaryFile(dir=output, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            torch.save({"identity": identity, "rows": rows}, temporary)
            temporary.replace(cache_path)
        finally:
            temporary.unlink(missing_ok=True)

    def cached(name, function):
        completed = rows.setdefault(name, {})

        def run(seed):
            if seed not in completed:
                completed[seed] = asdict(function(seed))
                if len(completed) % 16 == 0:
                    save_cache()
            return Result(**completed[seed])
        return run

    runners = {
        name: cached(name, lambda seed, head=head: run_episode(
            bundle, None, head, seed, config, limit=horizon
        ))
        for name, head in policies.items()
    }
    runners["random"] = cached("random", lambda seed: run_random(seed, config, limit=horizon))
    measured = evaluate(runners, seeds, config)
    save_cache()
    summary = {}
    for name, entry in measured.items():
        entry = dict(entry)
        entry.pop("episodes", None)
        summary[name] = entry
    comparison = summary["actor"]["versus_bc"]
    report = {
        "schema": "d4mj_lewm_execution_v1", "identity": identity,
        "primary": {
            "achievements_gap": comparison["achievements_gap"],
            "achievements_interval": comparison["achievements_interval"],
            "actor_beats_own_bc": comparison["achievements_beats"],
        },
        "policies": summary,
    }
    atomic_manifest(output / "evaluation.json", report)
    return report


def _interval(samples: torch.Tensor, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval, not a standard error or a Gaussian approximation."""
    tail = (1.0 - level) / 2
    ordered = samples.sort().values
    return (
        float(ordered[max(0, math.floor(tail * len(ordered)))]),
        float(ordered[min(len(ordered) - 1, math.ceil((1 - tail) * len(ordered)) - 1)]),
    )


def _result(seed, steps, total, terminated, truncated, env_state) -> Result:
    return Result(
        seed=seed,
        steps=steps,
        reward=total,
        terminated=terminated,
        truncated=truncated,
        achievements=tuple(bool(flag) for flag in np.asarray(env_state.achievements)),
    )
