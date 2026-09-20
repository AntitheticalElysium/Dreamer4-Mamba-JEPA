"""Stage 2/4: carry a jointly trained LeWM world into the agent surface.

Two phases, in the order the staged stop points read them:

  observed   the readout, behaviour cloning, reward and continuation heads are fitted on OBSERVED
             latents. A poor BC here is a representation/readout/data failure and nothing further
             is worth running.
  recursive  the same heads continue while the last `depth` readouts of every row come from
             GENERATED states, with depth raised H2 -> H16 on a linear schedule. A good BC with
             poor generated heads is a world/recursive-bridge failure.

What is reused rather than rebuilt: `Heads`, `head_targets`, `head_loss` and
`paired_terminal_loss` from `agent.py`, and `sample_batch`/`sample_terminal_batch` from `data.py`,
which carry the corrected relevant/uniform mixture, the event oversampling and the terminal
stratum. `train_agent` is NOT reused: it is Direct's loop, built on diffusion forcing, the
shortcut bootstrap and spatial slots, none of which LeWM has.

Three choices that are decisions, not defaults:

  frozen encoder   the representation is what the joint run produced; the bridge fits an agent on
                   top of it. Dreamer 4 freezes its tokenizer and continues dynamics while fitting
                   policy and reward heads; V-JEPA 2 freezes the encoder before training its
                   action-conditioned predictor. The world keeps training, which is what "continue
                   the same jointly trained world, do not restart Mamba" asks for.
  frozen predictor BN   `advance` refuses to stream under batch statistics, and it is right: the
                   deployed agent streams one state at a time and must use running statistics.
                   Training the world under statistics it will never see at deployment would be a
                   train/deploy mismatch, so the projector's BatchNorm is pinned for this stage.
  exact frames     `sample_batch` returns patches for the MAE encoder; LeWM wants raw uint8. The
                   round trip through `unpatchify` is exact and is asserted once at startup rather
                   than assumed.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads, head_loss, head_targets, paired_terminal_loss
from d4mj.config import Config, load_recipe
from d4mj.data import (Batch, EpisodeCorpus, atomic_manifest, load_joint_corpus, sample_batch,
                       sample_terminal_batch, unpatchify)
from d4mj.world_api import ModelBundle

CORPUS = [ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2"]


def agent_config(recipe) -> Config:
    """The legacy `Config` used ONLY for window geometry and batch stratification.

    `window=1` makes `receptive_field` 1 and `burn_in` 0, which is correct for a framewise ViT:
    the legacy 30-block burn-in exists for a temporal MAE encoder LeWM does not have.
    """
    a = recipe.agent
    return replace(Config(), window=1, batch=a.batch, sequence=a.sequence,
                   sequence_long=a.sequence * 4, dynamics_context=a.sequence * 3,
                   terminal_batch=a.terminal_batch, event_fraction=a.event_fraction,
                   direct_rollout=a.recursive_depth, bins=a.bins, mtp_leads=a.mtp_leads,
                   symlog_limit=a.symlog_limit, n_actions=recipe.dynamics.n_actions,
                   d_model=recipe.dynamics.width, seed=recipe.seed,
                   device=recipe.runtime.device)


def frames_of(batch: Batch, config: Config) -> torch.Tensor:
    """Patches back to native uint8 B,T,H,W,3. Exact: `patchify` only divides by 255."""
    pixels = unpatchify(batch.patches, config)
    return pixels.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()


def outgoing_actions(batch: Batch, n_actions: int) -> torch.Tensor:
    """Under led-to storage the action taken at block t is `led_to_action[t + 1]`.

    A pair objective supervises T-1 transitions, so the outgoing actions are blocks 1..T-1. Those
    are always real: block 0 is the only one whose incoming transition can be missing.
    """
    actions = batch.led_to_action[:, 1:]
    if bool(((actions < 0) | (actions >= n_actions)).any()):
        raise ValueError("a padding/BOS sentinel reached the outgoing action slice")
    return actions


def readouts(bundle, z, actions, depth: int):
    """Agent readouts for every block, the last `depth` of them from GENERATED states.

    Masking rather than slicing, for Direct's reason: every target keeps its own index, so the
    led-to offsets of the reward and policy leads are never re-derived against a shortened axis.
    """
    world = bundle.world
    blocks = z.shape[1]
    if depth <= 0:
        return world.teacher(z, actions).features, None
    if depth >= blocks:
        raise ValueError("a generated prefix must leave an observed block to start from")
    split = blocks - depth
    observed = world.teacher(z[:, :split], actions[:, :split - 1])
    state, features = observed.state, [observed.features]
    for offset in range(depth):
        state, feature = bundle.advance(state, actions[:, split - 1 + offset:split + offset])
        features.append(feature)
    mask = torch.zeros(blocks, dtype=torch.bool, device=z.device)
    mask[split:] = True
    return torch.cat(features, dim=1), mask


def move(batch: Batch, device: str) -> Batch:
    fields = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vars(batch).items()}
    return Batch(**fields)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a joint LeWM bundle")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", default=CORPUS)
    parser.add_argument("--head-steps", type=int, default=None)
    parser.add_argument("--recursive-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "d4mj_lewm_bundle_v2" or payload.get("phase") != "joint":
        raise SystemExit("bridge expects a LeWM joint bundle")
    recipe = load_recipe_from(payload)
    if recipe.agent is None:
        raise SystemExit("phase_gate: this checkpoint's recipe declares no agent; it is M0-M3")
    settings, cfg = recipe.agent, agent_config(recipe)
    device = recipe.runtime.device

    # The weights only. `restore_lewm_bundle` is a TRAINING RESUME: it reinstates optimizer,
    # sampler and RNG state and refuses a frozen encoder, none of which a bridge wants.
    bundle = ModelBundle.create(recipe)
    modules = payload["modules"]
    bundle.encoder.load_state_dict(modules["encoder"])
    bundle.world.load_state_dict(modules["world"])
    bundle.encoder.freeze()
    world = bundle.world
    world.train()
    world.agent_readout.requires_grad_(True)
    pinned = [m for m in world.predictor_projector.modules()
              if isinstance(m, torch.nn.BatchNorm1d)]
    for module in pinned:
        module.eval()

    episodes, contract = load_joint_corpus(args.corpus, recipe)
    # An EpisodeCorpus, not a list: it caches the per-length pools and draw tables. A plain list
    # sends `sample_batch` down its uncached path, which rescans every episode -- and calls
    # `events.any()` on each -- on every single step.
    keep = [i for i, e in enumerate(episodes) if e.split == "train"]
    train = episodes.subset(keep) if hasattr(episodes, "subset") else EpisodeCorpus(
        episodes[i] for i in keep)
    heads = Heads(recipe).to(device)
    parameters = [*world.parameters(), *heads.parameters()]
    optimizer = torch.optim.AdamW([p for p in parameters if p.requires_grad],
                                  lr=settings.learning_rate, weight_decay=1e-2)
    rng = torch.Generator().manual_seed(recipe.seed + 31)

    # A silent one-bit error in the round trip would corrupt every frame the agent ever sees.
    probe = sample_batch(train, torch.Generator().manual_seed(7), cfg, 0, 1, mixture=True)
    rebuilt = frames_of(probe, cfg)
    from d4mj.data import patchify
    if not torch.equal(patchify(rebuilt, cfg.patch), probe.patches):
        raise SystemExit("frame round trip is not exact; the agent would train on altered pixels")

    head_steps = args.head_steps if args.head_steps is not None else settings.head_steps
    recursive_steps = args.recursive_steps if args.recursive_steps is not None else settings.recursive_steps
    total = head_steps + recursive_steps
    atomic_manifest(args.out / "bridge.json", {
        "schema": "d4mj_m4_bridge_v1", "checkpoint": str(args.checkpoint),
        "variant": recipe.variant, "head_steps": head_steps, "recursive_steps": recursive_steps,
        "depth_schedule": [settings.recursive_depth, settings.recursive_depth_final],
        "frozen": ["encoder", "predictor_projector batchnorm"],
        "corpus": [str(p) for p in args.corpus],
        "bc_eligible_train_episodes": sum(1 for e in train if e.bc_eligible),
        "train_episodes": len(train),
        "sequence": cfg.sequence, "burn_in": cfg.burn_in})

    log, started = (args.out / "metrics.jsonl").open("a"), time.time()
    for step in range(total):
        recursive = step >= head_steps
        if recursive:
            share = (step - head_steps) / max(recursive_steps - 1, 1)
            depth = int(round(settings.recursive_depth
                              + share * (settings.recursive_depth_final - settings.recursive_depth)))
        else:
            depth = 0
        batch = move(sample_batch(train, rng, cfg, step, total, mixture=True), device)
        terminal = move(sample_terminal_batch(train, rng, cfg, step, total), device)
        losses = {}
        for name, source in (("main", batch), ("terminal", terminal)):
            with torch.no_grad():
                z = bundle.encoder(frames_of(source, cfg))
            actions = outgoing_actions(source, recipe.dynamics.n_actions)
            agent, mask = readouts(bundle, z, actions, min(depth, z.shape[1] - 1))
            readout = heads(agent) | {"centers": heads.centers}
            targets = head_targets(source, recipe)
            if name == "main":
                losses = head_loss(readout, targets, recipe)
                if mask is not None:
                    for key, value in head_loss(readout, targets, recipe, mask).items():
                        losses[f"generated_{key}"] = value
            else:
                losses["continuation"] = (
                    (1.0 - cfg.terminal_loss_mass) * losses["continuation"]
                    + cfg.terminal_loss_mass * paired_terminal_loss(readout, readout, targets))
        objective = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in parameters if p.requires_grad], 1.0)
        optimizer.step()

        row = {"step": step + 1, "phase": "recursive" if recursive else "observed", "depth": depth,
               "total": float(objective.detach()), "gradient_norm": float(norm),
               **{k: float(v.detach()) for k, v in losses.items()},
               "seconds": round(time.time() - started, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (step + 1) % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == total:
            torch.save({"world": world.state_dict(), "heads": heads.state_dict(),
                        "step": step + 1, "depth": depth, "variant": recipe.variant,
                        "config": payload["config"], "recipe_id": payload.get("recipe_id"),
                        "parent": str(args.checkpoint)},
                       args.out / f"bridge_{step + 1:06d}.pt")
    log.close()
    print(json.dumps({"status": "bridge_complete", "variant": recipe.variant, "steps": total}),
          flush=True)
    return 0


def load_recipe_from(payload):
    from d4mj.config import config_from_dict
    return config_from_dict(payload["config"])


if __name__ == "__main__":
    raise SystemExit(main())
