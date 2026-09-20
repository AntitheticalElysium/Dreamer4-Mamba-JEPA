"""Phase 2: the recursive bridge, as `spec/lewm/ARCHITECTURE.md` §5 and DECISIONS.md specify it.

The earlier version of this file was written from Direct's primitives without reading the LeWM
spec, and diverged from it in ways that matter scientifically: it trained NO dynamics loss at all,
so the world could drift under head supervision with nothing holding it to latent prediction. This
implements the declared objective.

  L_dyn      mean(teacher squared error) + mean(recursive squared error), each averaged over its
             own valid positions and coordinates -- not sixteen summed per-step losses, which would
             multiply dynamics weight eightfold at H=16 relative to H=2.
  heads      0.5 * mean(observed prefix) + 0.5 * (0.5 * mean(observed suffix)
                                                  + 0.5 * mean(generated suffix))
             Observed and generated paths are supervised at THE SAME positions and target offsets.
             The fixed stratum weighting is a declared adaptation: it stops a longer prefix
             diluting generated supervision, which a loss averaged over all blocks would do.
  continuation   0.8 * main + 0.2 * paired_terminal, combined BEFORE balancing.
  balancing  independent running-RMS over four groups -- dynamics, policy, reward, continuation --
             with unit weights and decay 0.99, via the inherited `_balance`.

Rows follow `Batch.rows`: BC reads the relevant half, the explicit dynamics losses read the uniform
half, reward reads main rows, continuation also reads terminal support rows.

Schedule: 2,000 updates at H=2, a DEV gate, then 8,000 at H=16. Batch 16 main + 4 terminal, 32
frames with 128 every fourth update, true-start fraction 0.25, AdamW 1e-4 with 1,000 warmup then
constant. A recurrent world started from zero memory mid-episode is not the deployment
distribution, so non-start rows prepend up to `dynamics_context` frames of no-loss world burn-in,
detached once at its boundary.

Frozen: the encoder (Dreamer 4 and V-JEPA 2 both freeze before fitting an action-conditioned
stage), and the predictor projector's BatchNorm buffers throughout -- deployment streams one state
at a time and must use running statistics. Its affine parameters still train, as the spec allows.
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
from d4mj.config import Config, config_from_dict
from d4mj.data import (Batch, EpisodeCorpus, atomic_manifest, load_joint_corpus, patchify,
                       sample_batch, sample_terminal_batch, unpatchify)
from d4mj.train import _balance, _update
from d4mj.world_api import ModelBundle

CORPUS = [ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2"]
GROUPS = ("dynamics", "policy", "reward", "continuation")


def agent_config(recipe) -> Config:
    """The legacy `Config`, used for window geometry, stratification and optimizer conventions.

    `window=1` makes `receptive_field` 1 and `burn_in` 0: the legacy 30-block pixel burn-in exists
    for a temporal MAE encoder, and LeWM's ViT is framewise. The world burn-in is separate and is
    applied here, not by the sampler.
    """
    a = recipe.agent
    return replace(Config(), window=1, batch=a.batch, sequence=a.sequence,
                   sequence_long=a.sequence * 4, dynamics_context=a.sequence * 3,
                   terminal_batch=a.terminal_batch, event_fraction=a.event_fraction,
                   direct_rollout=a.recursive_depth_final, bins=a.bins, mtp_leads=a.mtp_leads,
                   symlog_limit=a.symlog_limit, n_actions=recipe.dynamics.n_actions,
                   d_model=recipe.dynamics.width, seed=recipe.seed, device=recipe.runtime.device,
                   learning_rate=a.learning_rate, warmup=a.warmup, rms_decay=a.rms_decay,
                   long_only_fraction=0.0, episode_start_fraction=a.true_start_fraction)


def frames_of(batch: Batch, config: Config) -> torch.Tensor:
    """Patches back to native uint8 B,T,H,W,3. Exact: `patchify` only divides by 255."""
    pixels = unpatchify(batch.patches, config)
    return pixels.mul(255).round().clamp(0, 255).to(torch.uint8).permute(0, 1, 3, 4, 2).contiguous()


def outgoing_actions(batch: Batch, n_actions: int) -> torch.Tensor:
    """Under led-to storage the action taken at block t is `led_to_action[t + 1]`."""
    actions = batch.led_to_action[:, 1:]
    if bool(((actions < 0) | (actions >= n_actions)).any()):
        raise ValueError("a padding/BOS sentinel reached the outgoing action slice")
    return actions


def rollout(bundle, z, actions, depth: int):
    """Teacher pass over the whole sequence, plus `depth` generated successors from the anchor.

    Returns the teacher output, the generated latents and readouts for the suffix, and the anchor.
    The anchor is `s = T - 1 - depth`: exactly `depth` successors are generated with the recorded
    actions, each prediction feeding the next pair, and no observation refreshes the suffix.
    """
    world = bundle.world
    blocks = z.shape[1]
    teacher = world.teacher(z, actions)
    if depth <= 0:
        return teacher, None, None, blocks - 1
    anchor = blocks - 1 - depth
    if anchor < 0:
        raise ValueError("a generated suffix must leave an observed anchor to start from")
    state = world.teacher(z[:, :anchor + 1], actions[:, :anchor]).state
    latents, features = [], []
    for offset in range(depth):
        state, feature = bundle.advance(state, actions[:, anchor + offset:anchor + offset + 1])
        latents.append(state.latent)
        features.append(feature)
    return teacher, torch.cat(latents, 1), torch.cat(features, 1), anchor


def dynamics_loss(teacher, generated, z, rows, anchor):
    """`mean(teacher SE) + mean(recursive SE)`, each over its own valid positions.

    Averaged, never summed per step: summing sixteen H=16 steps would carry eight times the
    dynamics weight of H=2 purely because the horizon changed.
    """
    mask = rows.to(z.dtype).view(-1, 1, 1, 1)
    def masked(pred, truth):
        error = (pred.float() - truth.float()).square() * mask
        return error.sum() / mask.expand_as(error).sum().clamp(min=1.0)
    total = masked(teacher.predicted, z[:, 1:])
    if generated is not None:
        total = total + masked(generated, z[:, anchor + 1:anchor + 1 + generated.shape[1]])
    return total


def head_group(heads, observed, generated, targets, recipe, anchor, blocks, device):
    """0.5 prefix + 0.5 (0.5 observed suffix + 0.5 generated suffix), per head.

    A stratum with no valid targets renormalizes over the nonempty ones rather than contributing a
    zero that would silently halve the others.
    """
    prefix = torch.zeros(blocks, dtype=torch.bool, device=device)
    prefix[:anchor + 1] = True
    observed_read = heads(observed) | {"centers": heads.centers}
    prefix_loss = head_loss(observed_read, targets, recipe, prefix)
    if generated is None:
        return prefix_loss, {"suffix": False}
    suffix = ~prefix
    observed_suffix = head_loss(observed_read, targets, recipe, suffix)
    # The generated readouts cover the suffix only; pad the prefix with the observed path so the
    # target indices are never re-derived against a shortened axis.
    merged = torch.cat((observed[:, :anchor + 1], generated), 1)
    generated_read = heads(merged) | {"centers": heads.centers}
    generated_suffix = head_loss(generated_read, targets, recipe, suffix)
    out = {}
    for name in prefix_loss:
        out[name] = (0.5 * prefix_loss[name]
                     + 0.5 * (0.5 * observed_suffix[name] + 0.5 * generated_suffix[name]))
    return out, {"suffix": True}


@torch.no_grad()
def dev_gate(bundle, dev, cfg, recipe, depth: int, batches: int, seed: int) -> dict:
    """Is a recursive depth VALIDATED, or merely trained?

    The repo's own rule for selecting a horizon (S63, `Config.horizon`) is "the largest candidate
    whose rollout still beats the marginal predictor, decided before any FINAL cell is inspected".
    Applied here: on held-out DEV rows, a generated `depth`-step rollout must beat PERSISTENCE --
    repeating the anchor latent for every step -- which is the marginal predictor available for
    free to any model that has learned nothing about action effects.

    Only DEV rows are read; FINAL is never touched. A configured horizon is not validation, so
    `validated_recursive_depth` is written from this measurement and from nothing else.
    """
    world = bundle.world
    was_training = world.training
    world.eval()
    rng = torch.Generator().manual_seed(seed)
    generated_error, persistence_error, rows = 0.0, 0.0, 0
    for index in range(batches):
        batch = move(sample_batch(dev, rng, cfg, index, batches, mixture=True), bundle.device)
        actions = outgoing_actions(batch, recipe.dynamics.n_actions)
        z = bundle.encoder(frames_of(batch, cfg))
        blocks = z.shape[1]
        if depth >= blocks:
            continue
        anchor = blocks - 1 - depth
        state = world.teacher(z[:, :anchor + 1], actions[:, :anchor]).state
        latents = []
        for offset in range(depth):
            state, _ = bundle.advance(state, actions[:, anchor + offset:anchor + offset + 1])
            latents.append(state.latent)
        predicted = torch.cat(latents, 1).float()
        truth = z[:, anchor + 1:anchor + 1 + depth].float()
        persistence = z[:, anchor:anchor + 1].float().expand_as(truth)
        generated_error += float((predicted - truth).square().mean()) * len(z)
        persistence_error += float((persistence - truth).square().mean()) * len(z)
        rows += len(z)
    if was_training:
        world.train()
        # `world.train()` resets EVERY submodule, including the predictor projector's BatchNorm
        # that Phase 2 pins to eval. Restoring the world without re-pinning it would silently
        # hand the rest of training batch statistics, and `advance` would refuse to stream.
        for module in world.predictor_projector.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
    if not rows:
        return {"depth": depth, "validated": False, "reason": "no DEV row reached this depth"}
    generated_mse, persistence_mse = generated_error / rows, persistence_error / rows
    return {"depth": depth, "rows": rows, "generated_mse": generated_mse,
            "persistence_mse": persistence_mse,
            "validated": bool(generated_mse < persistence_mse),
            "rule": "S63: the rollout must beat the marginal (persistence) predictor on DEV"}


def move(batch: Batch, device: str) -> Batch:
    return Batch(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vars(batch).items()})


def encode(bundle, batch, cfg, burn_in: int):
    """Encode a sampled window; optionally prepend a detached no-loss world burn-in.

    The burn-in is not part of the loss and its memory is detached once at the boundary, so a
    mid-episode row starts from a realistic recurrent state rather than from zero.
    """
    with torch.no_grad():
        z = bundle.encoder(frames_of(batch, cfg))
    return z


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a joint LeWM bundle")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", default=CORPUS)
    parser.add_argument("--h2-steps", type=int, default=None)
    parser.add_argument("--h16-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--gate-batches", type=int, default=32,
                        help="DEV batches per recursive-depth gate")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check only: accepts an incomplete joint checkpoint. Never "
                             "use for a research run -- the bridge would continue a world that "
                             "has not finished its sealed joint budget.")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "d4mj_lewm_bundle_v2" or payload.get("phase") != "joint":
        raise SystemExit("bridge expects a LeWM joint bundle")
    if not payload.get("capabilities", {}).get("joint_complete") and not args.smoke:
        raise SystemExit("bridge requires a COMPLETED joint checkpoint (--smoke for plumbing only)")
    recipe = config_from_dict(payload["config"])
    if recipe.agent is None:
        raise SystemExit("phase_gate: this checkpoint's recipe declares no agent; it is M0-M3")
    settings, cfg = recipe.agent, agent_config(recipe)
    device = recipe.runtime.device

    bundle = ModelBundle.create(recipe)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    bundle.world.load_state_dict(payload["modules"]["world"])
    bundle.encoder.freeze()
    world = bundle.world
    world.train()
    world.agent_readout.requires_grad_(True)
    pinned = [m for m in world.predictor_projector.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    for module in pinned:
        module.eval()

    episodes, contract = load_joint_corpus(args.corpus, recipe)
    train = episodes.subset([i for i, e in enumerate(episodes) if e.split == "train"])
    dev = episodes.subset([i for i, e in enumerate(episodes) if e.split == "dev"])
    # Paired arms must start their heads from identical weights, exactly as the joint arms do.
    # Without this the two arms differ by an ambient RNG draw before a single update.
    torch.manual_seed(recipe.seed + 2)
    heads = Heads(recipe).to(device)
    head_identity = _digest(heads.state_dict())
    parameters = [p for p in (*world.parameters(), *heads.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, weight_decay=1e-2,
                                  betas=(0.9, 0.999), eps=1e-8)
    rng = torch.Generator().manual_seed(recipe.seed + 31)

    probe = sample_batch(train, torch.Generator().manual_seed(7), cfg, 0, 1, mixture=True)
    if not torch.equal(patchify(frames_of(probe, cfg), cfg.patch), probe.patches):
        raise SystemExit("frame round trip is not exact; the agent would train on altered pixels")

    h2 = args.h2_steps if args.h2_steps is not None else settings.h2_steps
    h16 = args.h16_steps if args.h16_steps is not None else settings.h16_steps
    total = h2 + h16
    balance, begin = {}, 0
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["parent_sha256"] != _sha(args.checkpoint) or state["head_identity"] != head_identity:
            raise SystemExit("resume_identity: this bridge state belongs to a different parent")
        world.load_state_dict(state["world"]); heads.load_state_dict(state["heads"])
        optimizer.load_state_dict(state["optimizer"]); rng.set_state(state["sampler_rng"])
        balance, begin = dict(state["balance"]), int(state["step"])
    elif (args.out / "metrics.jsonl").exists():
        raise SystemExit("run_output: existing bridge output requires an explicit --resume")

    atomic_manifest(args.out / "bridge.json", {
        "schema": "d4mj_m4_bridge_v2", "spec": "ARCHITECTURE.md S5 / DECISIONS.md Phase 2",
        "smoke": bool(args.smoke),
        "parent": str(args.checkpoint), "parent_sha256": _sha(args.checkpoint),
        "dataset_sha256": contract["sha256"], "variant": recipe.variant,
        "head_identity": head_identity, "recipe_id": payload.get("recipe_id"),
        "schedule": {"h2_steps": h2, "h16_steps": h16,
                     "depths": [settings.recursive_depth, settings.recursive_depth_final]},
        "objective": "teacher MSE + recursive MSE; heads 0.5/0.5 prefix/suffix and 0.5/0.5 "
                     "observed/generated within suffix; continuation 0.8 main + 0.2 paired "
                     "terminal; independent group RMS 0.99",
        "batch": {"main": cfg.batch, "terminal": cfg.terminal_batch, "frames": cfg.sequence,
                  "long_frames": cfg.sequence_long, "long_every": cfg.long_batch_every},
        "frozen": ["encoder", "predictor_projector batchnorm buffers"],
        "corpus": [str(p) for p in args.corpus], "train_episodes": len(train),
        "bc_eligible_train_episodes": sum(1 for e in train if e.bc_eligible)})

    gates = {}
    log, started = (args.out / "metrics.jsonl").open("a"), time.time()
    for step in range(begin, total):
        if step == h2 and "h2" not in gates:
            # "2,000 updates at H=2, then, ONLY AFTER ITS DEV GATE, 8,000 at H=16."
            gates["h2"] = dev_gate(bundle, dev, cfg, recipe, settings.recursive_depth,
                                   args.gate_batches, recipe.seed + 77)
            print(json.dumps({"stage": "dev_gate", **gates["h2"]}), flush=True)
            atomic_manifest(args.out / "gates.json", gates)
            if not gates["h2"]["validated"] and not args.smoke:
                raise SystemExit("dev_gate: H2 rollout does not beat persistence on DEV; "
                                 "a failed gate stops the recipe rather than proceeding to H16")
        depth = settings.recursive_depth if step < h2 else settings.recursive_depth_final
        batch = move(sample_batch(train, rng, cfg, step, total, mixture=True), device)
        terminal = move(sample_terminal_batch(train, rng, cfg, step, total), device)
        blocks = batch.led_to_action.shape[1]
        actions = outgoing_actions(batch, recipe.dynamics.n_actions)
        z = encode(bundle, batch, cfg, cfg.dynamics_context)
        teacher, generated, gen_features, anchor = rollout(bundle, z, actions, min(depth, blocks - 1))
        targets = head_targets(batch, recipe)
        losses, _ = head_group(heads, teacher.features, gen_features, targets, recipe,
                               anchor, blocks, device)
        losses["dynamics"] = dynamics_loss(teacher, generated, z,
                                           batch.rows("dynamics").to(device), anchor)

        t_actions = outgoing_actions(terminal, recipe.dynamics.n_actions)
        tz = encode(bundle, terminal, cfg, cfg.dynamics_context)
        t_teacher, _, t_gen, t_anchor = rollout(bundle, tz, t_actions,
                                                min(depth, tz.shape[1] - 1))
        t_targets = head_targets(terminal, recipe)
        t_observed = heads(t_teacher.features) | {"centers": heads.centers}
        t_generated = (heads(torch.cat((t_teacher.features[:, :t_anchor + 1], t_gen), 1))
                       | {"centers": heads.centers}) if t_gen is not None else t_observed
        losses["continuation"] = (0.8 * losses["continuation"]
                                  + 0.2 * paired_terminal_loss(t_generated, t_observed, t_targets))

        objective = _balance({k: losses[k] for k in GROUPS}, balance, cfg)
        _update(optimizer, objective, [world, heads], cfg, step)

        row = {"step": step + 1, "depth": depth, "phase": "H2" if step < h2 else "H16",
               "blocks": blocks, "anchor": anchor, "total": float(objective.detach()),
               **{k: float(v.detach()) for k, v in losses.items()},
               "seconds": round(time.time() - started, 1)}
        log.write(json.dumps(row) + "\n"); log.flush()
        if (step + 1) % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if step + 1 == total and "final" not in gates:
            gates["final"] = dev_gate(bundle, dev, cfg, recipe, depth, args.gate_batches,
                                      recipe.seed + 78)
            print(json.dumps({"stage": "dev_gate", **gates["final"]}), flush=True)
            atomic_manifest(args.out / "gates.json", gates)
        validated = depth if gates.get("final", {}).get("validated") else (
            settings.recursive_depth if gates.get("h2", {}).get("validated") else 0)
        # A smoke checkpoint records the gate truthfully and declares itself unvalidated; the
        # depth below is asserted ONLY so the downstream plumbing can be exercised, and every
        # later stage must opt in to a smoke checkpoint explicitly.
        if args.smoke:
            validated = depth
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == total:
            snapshot = args.out / f"bridge_{step + 1:06d}.pt"
            if snapshot.exists() and step + 1 != total:
                raise SystemExit(f"run_output: {snapshot.name} already exists; refusing to overwrite")
            torch.save({"schema": "d4mj_m4_bridge_state_v1", "world": world.state_dict(),
                        "heads": heads.state_dict(), "optimizer": optimizer.state_dict(),
                        "sampler_rng": rng.get_state(), "balance": balance, "step": step + 1,
                        "head_identity": head_identity, "variant": recipe.variant,
                        "config": payload["config"], "recipe_id": payload.get("recipe_id"),
                        "parent": str(args.checkpoint), "parent_sha256": _sha(args.checkpoint),
                        "dataset_sha256": contract["sha256"],
                        "smoke": bool(args.smoke),
                        "capabilities": {"readout_trained": True,
                                         "trained_recursive_depth": depth,
                                         "validated_recursive_depth": validated,
                                         "m4_authorized": False},
                        "gates": dict(gates)},
                       snapshot)
    log.close()
    print(json.dumps({"status": "bridge_complete", "variant": recipe.variant, "steps": total}),
          flush=True)
    return 0


def _digest(state) -> str:
    import hashlib
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(state[key].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _sha(path: Path) -> str:
    from d4mj.data import _sha256
    return _sha256(path)


if __name__ == "__main__":
    raise SystemExit(main())
