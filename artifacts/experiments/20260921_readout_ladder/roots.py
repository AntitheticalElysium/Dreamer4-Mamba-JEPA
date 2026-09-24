"""Seal a disjoint TRAIN/DEV root partition for the frozen readout ladder.

The gate's 512 H2 fork roots are left untouched: this reads which seeds they occupy and allocates
only from the remainder. Seeds allocated here are RETIRED from future evaluation use, because a
head fitted on them has seen their outcomes.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.config import config_from_dict
from d4mj.data import _sha256
from d4mj.lewm_diagnostics import FORK_STORE, fork_population


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/lewm_m4_canonical/raw")
    parser.add_argument("--eval-roots", type=int, default=512)
    parser.add_argument("--train-seeds", type=int, default=700)
    parser.add_argument("--dev-seeds", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--out", type=Path, default=HERE / "evidence")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    recipe = config_from_dict(json.loads((args.run / "resolved_recipe.json").read_text()))
    held = fork_population(recipe, roots=args.eval_roots, seed=recipe.seed + 51)
    reserved = sorted(set(held["seed"].tolist()))

    every = sorted({int(Path(p).stem.split("-")[1])
                    for p in glob.glob(str(FORK_STORE / "seed-*.pt"))})
    free = [s for s in every if s not in set(reserved)]
    order = torch.randperm(len(free), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train = sorted(free[i] for i in order[:args.train_seeds])
    dev = sorted(free[i] for i in order[args.train_seeds:args.train_seeds + args.dev_seeds])
    spare = sorted(set(free) - set(train) - set(dev))

    for left, right, name in ((reserved, train, "eval/train"), (reserved, dev, "eval/dev"),
                              (train, dev, "train/dev")):
        if set(left) & set(right):
            raise SystemExit(f"partition overlap in {name}")

    ledger = {"schema": "d4mj_ladder_roots_v1",
              "store": str(FORK_STORE), "store_manifest_absent": True,
              "reserved_for_gate": {"seeds": reserved, "count": len(reserved),
                                    "roots": held["roots"],
                                    "note": "the sealed H2 gate population; untouched here"},
              "fit_train": {"seeds": train, "count": len(train)},
              "fit_dev": {"seeds": dev, "count": len(dev)},
              "unallocated": {"count": len(spare),
                              "note": "still available for future evaluation"},
              "retirement": "fit_train and fit_dev seeds are RETIRED from future evaluation: a "
                            "head fitted or selected on them has seen their outcomes",
              "partition_seed": args.seed}
    path = args.out / "root_partition.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({"reserved": len(reserved), "train_seeds": len(train),
                      "dev_seeds": len(dev), "unallocated": len(spare),
                      "ledger_sha256": _sha256(path)[:16]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
