"""Stage 4/5: the primary result -- each actor against its OWN BC, on real Craftax.

Protocol copied from `eda/run_paired_execution.py`, which produced the Direct table this campaign
is measured against, so the two are directly comparable rather than merely similar:

  seeds     30000..30511, 512 shared DEV seeds, every policy on the same ones
  horizon   the native 10000-step Craftax cap, not the collector's 2500
  sampling  categorical at temperature 1; greedy is a declared secondary, not the headline
  interval  `execution.evaluate`'s paired episode bootstrap, reporting the achievement gap and
            whether its 95% interval excludes zero

The BC arm is the FROZEN PRIOR saved beside the actor -- the same heads the actor was initialized
from and KL-bounded against. That is what makes it "its own BC" rather than a different model's.

Every executed episode is cached by (policy, seed), so a 512-seed run resumes instead of
re-executing; Direct's took 167.8 minutes.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.config import config_from_dict
from d4mj.data import atomic_manifest
from d4mj.execution import Result, evaluate, run_episode, run_random
from d4mj.world_api import ModelBundle

SEED_BASE = 30_000


def policies_from(actor_path: Path, joint_path: Path, device: str):
    """The actor and the frozen prior it was trained from, sharing one world and encoder."""
    payload = torch.load(actor_path, map_location="cpu", weights_only=False)
    recipe = config_from_dict(payload["config"])
    if recipe.agent is None:
        raise SystemExit("phase_gate: this checkpoint's recipe declares no agent")
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
    bundle = ModelBundle.create(recipe)
    bundle.encoder.load_state_dict(joint["modules"]["encoder"])
    bundle.world.load_state_dict(payload["world"])
    # Carry the actor's capability record: execution is control, and control is authorized by a
    # recorded capability, never by the recipe that configured it.
    bundle.capabilities = dict(payload["capabilities"])
    bundle.eval()
    heads = {}
    for name, key in (("actor", "heads"), ("bc", "prior")):
        head = Heads(recipe).to(device)
        head.load_state_dict(payload[key])
        heads[name] = head.eval()
    return bundle, heads, recipe, payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True,
                        help="the joint bundle supplying the frozen encoder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed-base", type=int, default=SEED_BASE)
    parser.add_argument("--limit", type=int, default=None,
                        help="step cap; the default is the recipe's native horizon_eval")
    parser.add_argument("--random-control", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true",
                        help="plumbing check only: accepts a smoke actor, whose horizon was "
                             "asserted rather than validated")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    probe = config_from_dict(torch.load(args.actor, map_location="cpu", weights_only=False)["config"])
    device = probe.runtime.device
    bundle, heads, recipe, payload = policies_from(args.actor, args.joint, device)
    if payload.get("smoke") and not args.smoke:
        raise SystemExit("phase_gate: this actor is a smoke artifact and its numbers are "
                         "meaningless; pass --smoke only to exercise plumbing")
    episodes = args.episodes if args.episodes is not None else recipe.agent.eval_episodes
    seeds = list(range(args.seed_base, args.seed_base + episodes))

    # Cached executions are bound to everything that determines them. Reusing a directory with a
    # different actor, world, encoder, horizon or seed base would otherwise silently report old
    # episodes as if they came from the new model.
    from d4mj.data import _sha256
    identity = {"schema": "d4mj_m4_execution_identity_v1",
                "actor_sha256": _sha256(args.actor), "joint_sha256": _sha256(args.joint),
                "executed_horizon": payload.get("executed_horizon"),
                "variant": recipe.variant, "seed_base": args.seed_base,
                "limit": args.limit or recipe.horizon_eval,
                "protocol": "eda/run_paired_execution.py; categorical temperature 1"}
    store = args.out / "episodes.pt"
    cached = torch.load(store, weights_only=False) if store.exists() else {}
    if cached and cached.get("identity") != identity:
        raise SystemExit("execution_cache: this directory holds episodes executed under a "
                         "different actor/world/horizon/protocol; use a fresh --out")
    rows = cached.get("rows", {}) if cached else {}

    def cached(name, run):
        done = rows.setdefault(name, {})

        def episode(seed: int) -> Result:
            if seed not in done:
                done[seed] = asdict(run(seed))
                if len(done) % 16 == 0:
                    torch.save({"identity": identity, "rows": rows}, store)
                    print(json.dumps({"policy": name, "done": len(done),
                                      "of": len(seeds)}), flush=True)
            return Result(**done[seed])
        return episode

    runners = {name: cached(name, (lambda h: lambda seed: run_episode(
        bundle, None, h, seed, recipe, limit=args.limit))(head))
        for name, head in heads.items()}
    if args.random_control:
        runners["random"] = cached("random", lambda seed: run_random(seed, recipe, limit=args.limit))

    started = time.time()
    print(json.dumps({"stage": "execute", "policies": sorted(runners), "seeds": len(seeds),
                      "horizon": args.limit or recipe.horizon_eval}), flush=True)
    report = evaluate(runners, seeds, recipe)
    torch.save({"identity": identity, "rows": rows}, store)

    summary = {}
    for name, entry in report.items():
        entry.pop("episodes", None)
        summary[name] = entry
    contest = summary.get("actor", {}).get("versus_bc", {})
    result = {"schema": "d4mj_m4_evaluation_v2", "identity": identity, "variant": recipe.variant,
              "actor": str(args.actor), "seed_base": args.seed_base, "episodes": len(seeds),
              "horizon": args.limit or recipe.horizon_eval,
              "protocol": "eda/run_paired_execution.py, same 512 DEV seeds and native cap",
              "primary": {"achievements_gap": contest.get("achievements_gap"),
                          "achievements_interval": contest.get("achievements_interval"),
                          "actor_beats_own_bc": contest.get("achievements_beats")},
              "policies": summary, "seconds": round(time.time() - started, 1)}
    atomic_manifest(args.out / "evaluation.json", result)
    print(json.dumps({"status": "evaluation_complete", "variant": recipe.variant,
                      **result["primary"]}), flush=True)
    for name in ("bc", "actor", "random"):
        if name in summary:
            e = summary[name]
            print(f"  {name:8s} achieve {e['achievements']:.3f}  score {e['score']:.3f}  "
                  f"reward {e['reward']:.3f}  length {e['length']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
