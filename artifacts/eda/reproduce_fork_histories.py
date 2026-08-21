"""Latent history at the fixed fork roots, for evaluating a history-conditioned model.

The 965 branched roots and the 104 policy forks store only (seed, step) and their
simulator truth. Both collections were produced by the same frozen BC policy under
checkpoints that still hash to their recorded digests, so the rollouts are replayed
and the committed latent history kept at each fork root. Reproduction is checked
against each collection's own recorded trajectory action.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from artifacts.localize_counterfactual import load_models
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

PHASE1A = ROOT / "artifacts/stage_a_terminalfix/phase1a.pt"
PHASE2 = ROOT / "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"
HISTORY = 64
OUT = HERE / "fork_histories"
OUT.mkdir(exist_ok=True)

base = Config()
config = Config(transition="direct", time_mixer="attention")
encoder, world, heads = load_models(PHASE1A, PHASE2, base, config)

branched = torch.load(ROOT / "artifacts/branched_coverage_gate/branched_forks.pt",
                      weights_only=False)
policy = torch.load(
    ROOT / "artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/"
    "paired_trajectory_forks.pt", weights_only=False)

SMOKE = "--smoke" in sys.argv
JOBS = {
    "branched_965": (branched, 14_000, 8 if SMOKE else 512, 400),
    "policy_fork_104": (policy, 13_000, 8 if SMOKE else 128, 400),
}


@torch.no_grad()
def run(name, saved, seed_start, seed_count, limit):
    wanted: dict[int, set[int]] = {}
    index_of: dict[tuple[int, int], int] = {}
    for row, (s, t) in enumerate(zip(saved["seed"].tolist(), saved["step"].tolist())):
        wanted.setdefault(s, set()).add(t)
        index_of[(s, t)] = row
    records: list[dict] = []
    matched = 0
    started = time.time()
    for order, seed in enumerate(range(seed_start, seed_start + seed_count)):
        if seed not in wanted:
            continue
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full((1, 1), config.n_actions, dtype=torch.long,
                              device=config.device)
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
        latents: list[torch.Tensor] = []
        frames: list[torch.Tensor] = []
        # led-to convention: `led[j]` is the action that produced observation j,
        # BOS at a true episode start -- the same array `data._window` builds.
        led: list[int] = [config.n_actions]
        target = max(wanted[seed])
        for index in range(min(target + 1, limit)):
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(world, encoder, state, incoming, patches,
                                   world_rng, config)
            latents.append(state.world.latent[0, -1].cpu())
            frames.append(observation.clone())
            logits = heads(agent)["policy"][:, -1, 0]
            chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))
            if index in wanted[seed]:
                row = index_of[(seed, index)]
                if int(saved["trajectory_action"][row]) != chosen:
                    raise AssertionError(f"{name}: action mismatch at {seed}:{index}")
                matched += 1
                history = torch.stack(latents[-HISTORY:])
                records.append(dict(
                    row=row, seed=seed, step=index,
                    history=history,
                    frames=torch.stack(frames[-HISTORY:]),
                    led_to_action=torch.tensor(led[-len(history):], dtype=torch.long),
                    true_death=saved["true_death"][row],
                    trajectory_action=chosen,
                ))
            observation, env_state, _, terminated, truncated = env_step(
                env_state, chosen, seed + index + 1)
            led.append(chosen)
            incoming.fill_(chosen)
            if terminated or truncated:
                break
        if (order + 1) % 50 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  {name}: seed {order + 1}/{seed_count} "
                  f"[{time.time()-started:.0f}s, {(seed_count-order-1)/rate:.0f}s left]",
                  flush=True)
    if not SMOKE and matched != len(saved["seed"]):
        raise AssertionError(f"{name}: reproduced {matched} of {len(saved['seed'])} roots")
    torch.save(records, OUT / (f"{name}.smoke.pt" if SMOKE else f"{name}.pt"))
    print(f"{name}: reproduced and verified {matched} roots", flush=True)


for name, (saved, seed_start, seed_count, limit) in JOBS.items():
    run(name, saved, seed_start, seed_count, limit)
(OUT / "manifest.json").write_text(json.dumps(
    {"history": HISTORY, "phase1a": str(PHASE1A), "phase2": str(PHASE2)}, indent=2))
print("fork histories complete")
