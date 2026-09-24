"""Run the H2/H16/actor gate with the model in the eval mode a gate must measure.

`python -m d4mj gate` builds its bundle with `ModelBundle.create`, whose modules default to
TRAINING mode, and never switches them. Two things then go wrong, and the second is fatal:

  * the encoder's projector BatchNorm uses BATCH statistics, so a measurement depends on how the
    gate happened to chunk its input -- which is exactly the batch-invariance property the cache
    export refuses to ship without;
  * with 129 retention addresses and a chunk of 32 the last chunk holds one frame, and BatchNorm1d
    raises "Expected more than 1 value per channel when training".

The fix belongs in `_load_bridge_parent`, but `experiments.py` and `lewm_diagnostics.py` are both
inside the sealed source closure: editing either orphans the joint and bridge checkpoints and
costs a full re-run. This runner lives outside the closure, calls the same functions, and sets the
eval mode the gate should always have used.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT))

from d4mj.config import config_from_dict, load_recipe
from d4mj.data import load_joint_corpus
from d4mj.experiments import (_cache_from_contract, _load_actor_parent, _load_bridge_parent)
from d4mj.lewm_diagnostics import actor_gate, bridge_gate, fork_population, retention_addresses


def frozen(bundle, *modules):
    """Everything the gate measures is frozen: encoder exported, world streaming, BN fixed."""
    bundle.encoder.freeze()
    bundle.world.eval()
    for module in bundle.world.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    for extra in modules:
        if extra is not None:
            extra.eval()
    return bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("h2", "h16", "actor"), required=True)
    parser.add_argument("--dataset", type=Path, nargs="+")
    parser.add_argument("--fork-roots", type=int, default=512)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)

    recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
    settings = recipe.agent
    screen = load_recipe(ROOT / "d4mj/recipes/joint_screen.json")
    cache_path = args.run / "cache"
    output = args.run / "gates" / args.stage
    panel = retention_addresses()
    forks = fork_population(recipe, roots=args.fork_roots, seed=recipe.seed + 51)
    reference = None
    if args.reference and args.reference.is_file():
        stored = torch.load(args.reference, map_location="cpu", weights_only=False)
        from d4mj.world_api import ModelBundle
        reference = ModelBundle.create(config_from_dict(stored["config"]))
        reference.encoder.load_state_dict(stored["modules"]["encoder"], strict=True)
        reference.encoder.freeze()

    if args.stage == "actor":
        checkpoint = args.run / "actor" / f"step-{settings.actor_screen_steps:06d}.pt"
        bundle, heads, prior, payload = _load_actor_parent(checkpoint)
        frozen(bundle, heads, prior)
        episodes, contract = _cache_from_contract(cache_path, bundle, payload["cache"])
        report = actor_gate(bundle, heads, prior, payload, episodes, contract, screen, output,
                            checkpoint=checkpoint, forks=forks)
    else:
        steps = settings.h2_steps if args.stage == "h2" else settings.h2_steps + settings.h16_steps
        checkpoint = args.run / "bridge" / f"step-{steps:06d}.pt"
        bundle, heads, payload = _load_bridge_parent(checkpoint)
        frozen(bundle, heads)
        episodes, contract = _cache_from_contract(cache_path, bundle, payload["cache"])
        raw = load_joint_corpus(args.dataset, recipe)[0] if args.dataset else None
        report = bridge_gate(bundle, heads, payload, episodes, contract, screen, output,
                             stage=args.stage, checkpoint=checkpoint, raw_episodes=raw,
                             forks=forks, panel=panel, reference=reference)
    failed = sorted(n for n, c in report["components"].items() if c["status"] != "pass")
    print(json.dumps({"stage": args.stage, "decision": report["decision"],
                      "validated_recursive_depth": report["validated_recursive_depth"],
                      "failed_components": failed}), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
