"""Stage 4: train the imagination actor, mirroring Phase 3 for a LeWM world.

The structure is `train.train_actor`'s, reproduced rather than called because that function builds
Direct's committed/conditioning inputs and a legacy `WorldState`. Everything that makes Phase 3 a
valid experiment is kept:

  frozen world     the actor may not improve by reshaping the model it is scored inside.
  frozen prior     the behaviour-cloned heads are deep-copied and frozen; `prior_beta` bounds how
                   far the policy may move from them in reverse KL.
  frozen model body   reward and continuation freeze with the world, so the actor cannot move its
                   own reward signal. Only `heads.actor_parameters()` -- policy, critic and their
                   separate bodies -- receive gradient.
  mixture starts   imagined trajectories begin from the same relevant/uniform mixture BC used, so
                   RL does not start away from the events carrying the sparse reward.

S68, restated for this architecture: the actor may not imagine past the depth the bridge actually
trained. Direct checks `horizon <= direct_rollout`; here the bound is `min(trained_recursive_depth,
validated_recursive_depth)` read from the bridge checkpoint's capability record -- a configured
horizon is not validation.

Optimization conventions are Direct's, because a difference from Direct that came from the
optimizer rather than from LeWM/Mamba would not answer the question this campaign asks:

  balancing   independent running-RMS over the actor and critic groups, decay 0.99, unit weights
  schedule    AdamW 1e-4, 1,000 warmup updates then constant
  budget      DECISIONS.md "Phase 3": 16 starting contexts, 500-update screen, 5,000 total
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.actor_critic import actor_loss, critic_loss, lambda_returns
from d4mj.agent import Heads
from d4mj.config import config_from_dict
from d4mj.data import _sha256, atomic_manifest, load_joint_corpus, sample_batch
from d4mj.imagination import imagine
from d4mj.train import _balance, _update
from d4mj.world_api import ModelBundle

from bridge import agent_config, frames_of, move, outgoing_actions

CORPUS = [ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2"]


def load_bridge(path: Path, device: str):
    """A bridge checkpoint: the continued world, the fitted heads, and its capability record."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "d4mj_m4_bridge_state_v1":
        raise SystemExit("actor expects a bridge state checkpoint")
    recipe = config_from_dict(payload["config"])
    if recipe.agent is None:
        raise SystemExit("phase_gate: this checkpoint's recipe declares no agent")
    bundle = ModelBundle.create(recipe)
    bundle.world.load_state_dict(payload["world"])
    bundle.capabilities = dict(payload["capabilities"])
    heads = Heads(recipe).to(device)
    heads.load_state_dict(payload["heads"])
    return bundle, heads, recipe, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True,
                        help="the joint bundle the bridge continued; supplies the frozen encoder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", default=CORPUS)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check only: accepts a smoke bridge checkpoint, whose "
                             "recursive depth was asserted rather than validated on DEV")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    joint = torch.load(args.encoder_checkpoint, map_location="cpu", weights_only=False)
    device = config_from_dict(joint["config"]).runtime.device
    bundle, heads, recipe, payload = load_bridge(args.bridge, device)
    if payload.get("smoke") and not args.smoke:
        raise SystemExit("phase_gate: this bridge checkpoint is a smoke artifact; its recursive "
                         "depth was asserted, not validated. Pass --smoke to exercise plumbing.")
    if payload["parent_sha256"] != _sha256(args.encoder_checkpoint):
        raise SystemExit("identity: this bridge did not continue the supplied joint checkpoint")
    bundle.encoder.load_state_dict(joint["modules"]["encoder"])
    bundle.encoder.freeze()
    settings = recipe.agent
    # The bound is what the bridge RECORDED, not what the recipe configured.
    trained = payload["capabilities"].get("trained_recursive_depth", 0)
    validated = payload["capabilities"].get("validated_recursive_depth", 0)
    horizon = min(settings.horizon, trained, validated) if validated else min(settings.horizon, trained)
    if horizon < 1:
        raise SystemExit("phase_gate: the bridge records no trained recursive depth")
    if validated == 0:
        print(json.dumps({"warning": "validated_recursive_depth is 0; this actor run is a "
                                     "DIAGNOSTIC at the trained depth, not a validated horizon",
                          "trained": trained, "horizon": horizon}), flush=True)
    # Phase 3 freezes the entire world and readout in eval mode, buffers included.
    bundle.world.eval()
    for parameter in bundle.world.parameters():
        parameter.requires_grad_(False)
    prior = copy.deepcopy(heads).eval()
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.parameters():
        parameter.requires_grad_(False)
    for parameter in heads.actor_parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(heads.actor_parameters(), lr=settings.learning_rate,
                                  weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8)

    episodes, contract = load_joint_corpus(args.corpus, recipe)
    keep = [i for i, e in enumerate(episodes) if e.split == "train"]
    train = episodes.subset(keep)
    cfg = agent_config(recipe)
    from dataclasses import replace
    sampling = replace(cfg, batch=settings.actor_batch)
    rng = torch.Generator().manual_seed(recipe.seed + 3)
    policy_rng = torch.Generator(device=device).manual_seed(recipe.seed + 2**20)
    steps = args.steps if args.steps is not None else settings.actor_steps
    horizon_recipe = replace(recipe, agent=replace(settings, horizon=horizon))
    balance, begin = {}, 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["bridge_sha256"] != _sha256(args.bridge):
            raise SystemExit("resume_identity: this actor state belongs to a different bridge")
        heads.load_state_dict(state["heads"]); optimizer.load_state_dict(state["optimizer"])
        rng.set_state(state["sampler_rng"]); policy_rng.set_state(state["policy_rng"].to(policy_rng.get_state().device) if hasattr(state["policy_rng"], "to") else state["policy_rng"])
        balance, begin = dict(state["balance"]), int(state["step"])
    elif (args.out / "metrics.jsonl").exists():
        raise SystemExit("run_output: existing actor output requires an explicit --resume")

    atomic_manifest(args.out / "actor.json", {
        "schema": "d4mj_m4_actor_v2", "bridge": str(args.bridge),
        "bridge_sha256": _sha256(args.bridge), "parent_sha256": payload["parent_sha256"],
        "dataset_sha256": contract["sha256"], "variant": recipe.variant, "steps": steps,
        "screen_steps": settings.actor_screen_steps,
        "configured_horizon": settings.horizon, "trained_recursive_depth": trained,
        "validated_recursive_depth": validated, "executed_horizon": horizon,
        "optimizer": "AdamW 1e-4, decay .01, warmup 1000 then constant; actor/critic RMS 0.99",
        "frozen": ["encoder", "world (eval, buffers included)", "reward/continuation body",
                   "prior policy"],
        "trainable": "policy, critic and their separate bodies"})

    log, started = (args.out / "metrics.jsonl").open("a"), time.time()
    for step in range(steps):
        batch = move(sample_batch(train, rng, sampling, step, steps, mixture=True), device)
        actions = outgoing_actions(batch, recipe.dynamics.n_actions)
        with torch.no_grad():
            z = bundle.encoder(frames_of(batch, cfg))
            observed = bundle.world.teacher(z, actions)
        trajectory = imagine(bundle, heads, observed.state, observed.features[:, -1:],
                             None, policy_rng, horizon_recipe)
        returns = lambda_returns(trajectory, recipe)
        with torch.no_grad():
            reference = prior(trajectory.agent[:, :-1])["policy"][:, :, 0]
        losses = {"actor": actor_loss(trajectory, returns, reference, recipe),
                  "critic": critic_loss(heads(trajectory.agent[:, :-1])["value"], returns,
                                        heads.centers)}
        # Direct's conventions: independent running-RMS per group, then warmup-then-constant.
        objective = _balance(losses, balance, cfg)
        _update(optimizer, objective, [heads], cfg, step)

        row = {"step": step + 1, "screen": step + 1 <= settings.actor_screen_steps,
               "horizon": horizon, **{k: float(v.detach()) for k, v in losses.items()},
               "total": float(objective.detach()),
               "imagined_reward": float(trajectory.reward.mean()),
               "imagined_continuation": float(trajectory.continuation.mean()),
               "seconds": round(time.time() - started, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if (step + 1) % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == steps:
            snapshot = args.out / f"actor_{step + 1:06d}.pt"
            if snapshot.exists() and step + 1 != steps:
                raise SystemExit(f"run_output: {snapshot.name} already exists; refusing to overwrite")
            torch.save({"schema": "d4mj_m4_actor_state_v1",
                        "heads": heads.state_dict(), "prior": prior.state_dict(),
                        "world": bundle.world.state_dict(), "optimizer": optimizer.state_dict(),
                        "sampler_rng": rng.get_state(), "policy_rng": policy_rng.get_state(),
                        "balance": balance, "step": step + 1, "executed_horizon": horizon,
                        "variant": recipe.variant, "config": payload["config"],
                        "recipe_id": payload.get("recipe_id"),
                        "bridge": str(args.bridge), "bridge_sha256": _sha256(args.bridge),
                        "parent_sha256": payload["parent_sha256"],
                        "dataset_sha256": contract["sha256"],
                        "smoke": bool(args.smoke or payload.get("smoke")),
                        "capabilities": dict(payload["capabilities"],
                                             validated_recursive_depth=horizon,
                                             m4_authorized=True)},
                       snapshot)
    log.close()
    print(json.dumps({"status": "actor_complete", "variant": recipe.variant, "steps": steps}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
