"""Second world seed for the representation-interface comparison, judged on a fresh block.

INTERFACE.md (seed 1, sealed 55k): patch_interface_better, u_world_state_passes, and the post-hoc
shuffle control found U's generated state alone ranks actions (state_alone_carries_consequence) while
Z's token probe was a root-plus-action shortcut. Both trained systems failed. One world seed, and the 55k
block has now been read twice. This retrains both arms with a second, independent seed and judges them,
and the seed-1 worlds, once on a NEW block.

Everything is `interface.py` (committed recipe, unchanged code path) except the seeds and paths:
  seed 2    world init 8, phase-1 batch order 12, head init config.seed + 3, phase-2 batch order 18,
            generation-depth draws 14 (seed 1: 7 / 11 / +2 / 17 / 13). The pool, its PCA, the loss
            weights and every schedule are shared.
  worlds    artifacts/eda/interface_worlds_v2 (seed 2); seed-1 worlds reused from interface_worlds_v1
  judge     `observe.py collect --seed-start 56000 --target-opportunity 800 --max-seeds 1500 --out
            artifacts/eda/observe_fresh_v7`, collected after this commit, read once per run below
Runs on the new block, each with `interface.score` (all interface.py rules and probes) then
`interface_shuffle.main` (token and token-free probes, intact / within-root permuted / mean):
  seed 2 -> evidence/interface_seed2_56k.json, interface_shuffle_seed2_56k.json    (the replication)
  seed 1 -> evidence/interface_seed1_56k.json, interface_shuffle_seed1_56k.json    (a fresh look at seed 1)

DECLARED READING, seed 2 (committed before collection and before seed-2 training):
  replicates        interface = patch_interface_better AND world = u_world_state_passes AND the shuffle
                    control's U reading = state_alone_carries_consequence (P0 ok)
  partial           two of the three
  does_not_replicate  otherwise (or P0 void)
Declared prediction, unchanged from interface.py: trained_system_fails in both arms, both seeds.
Reported: the same readings for seed 1 on this block; every contrast for both seeds side by side.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

import interface as I  # noqa: E402
import interface_shuffle as S  # noqa: E402

SEED2 = {"init": 8, "phase1": 12, "heads": 3, "phase2": 18, "depth": 14}
WORLDS = {1: ROOT / "artifacts/eda/interface_worlds_v1", 2: ROOT / "artifacts/eda/interface_worlds_v2"}
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"


def configure(seed):
    I.SEEDS = dict(SEED2) if seed == 2 else I.SEEDS
    I.WORLDS, I.SEALED = WORLDS[seed], JUDGE
    I.MIN_JUDGE_SEED, I.USED_STORES = 56_000, range(1, 7)
    I.EVIDENCE = f"interface_seed{seed}_56k.json"
    S.RECORD, S.OUT = I.EVIDENCE, f"interface_shuffle_seed{seed}_56k.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "decide"))
    parser.add_argument("--arm", choices=I.ARMS)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    configure(args.seed)
    if args.command == "train":
        if args.seed != 2:
            raise SystemExit("only seed 2 is trained here")
        pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
        world, heads, history, counts = I.train(args.arm, pool, device, log)
        I.WORLDS.mkdir(parents=True, exist_ok=True)
        torch.save({"arm": args.arm, "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                    "depth_counts": counts, "seeds": I.SEEDS, "script_sha256": _sha256(HERE / "interface.py"),
                    "replicate_sha256": _sha256(Path(__file__)),
                    "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, I.WORLDS / f"{args.arm}.pt")
        log(status="train_complete", arm=args.arm, depth_counts=counts)
    elif args.command == "score":
        I.score(device, log)
        S.main()
    else:
        out = {}
        for seed in (1, 2):
            a = json.loads((HERE / f"evidence/interface_seed{seed}_56k.json").read_text())["readings"]
            b = json.loads((HERE / f"evidence/interface_shuffle_seed{seed}_56k.json").read_text())["readings"]
            hits = [a["interface"] == "patch_interface_better", a["world"] == "u_world_state_passes",
                    b["U"] == "state_alone_carries_consequence"]
            reading = ("does_not_replicate" if a["P0"] != "ok" else "replicates" if all(hits) else
                       "partial" if sum(hits) == 2 else "does_not_replicate")
            out[f"seed{seed}"] = {"reading": reading, "interface": a, "shuffle": b}
        (HERE / "evidence/replicate_56k.json").write_text(json.dumps(
            {"schema": "d4mj_replicate_v1", "script_sha256": _sha256(Path(__file__)), **out}, indent=2) + "\n")
        log(status="replicate_complete", seed2=out["seed2"]["reading"], seed1=out["seed1"]["reading"])


if __name__ == "__main__":
    raise SystemExit(main())
