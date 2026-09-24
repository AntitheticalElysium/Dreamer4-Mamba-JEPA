"""Stage 1: the merged world-training corpus -- the expert archive beside support-v2.

The first paired run saw support-v2 alone: 8,069 TRAIN episodes, `bc_eligible: false`, so it had
no behaviour-cloning data at all. Direct trained on the expert archive plus support-v2. Matching
that is the point of this campaign, so the corpus has to be matched before anything else is.

The two sources keep their own identities rather than being flattened into one:

  splits       the archive splits by `episode_splits(len(archive), Config().seed)`, exactly as
               `eda/corpus.py` does, in the same permutation order; support-v2 keeps its on-disk
               `sha256(seed:round:slot)` 80/10/10 field. Neither is re-derived, so no episode can
               silently change side and no DEV/FINAL frame can enter training.
  eligibility  the archive is BC-eligible, support-v2 is not. Both are uniform-eligible, so both
               feed the world; only the archive feeds behaviour cloning. The primary result is
               "actor versus its OWN BC", so that asymmetry is recorded here rather than
               discovered later.

Only the archive is converted, because it is a bare list of records rather than an episode store.
Support-v2 is referenced where it already lives: `load_joint_corpus` pins each source by its own
manifest digest, so the merged corpus is immutable by reference and nothing is copied twice.
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

from d4mj.config import Config
from d4mj.data import STORE_FORMAT, atomic_manifest, save_episode_shard
from d4mj.expert import load_archive
from d4mj.data import episode_splits

ARCHIVE = ROOT / "d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt"
SUPPORT = ROOT / "artifacts/craftax_support_v2"


def merge(out: Path, per_shard: int, limit: int | None) -> dict:
    """Convert the archive into an episode store, then declare the two-source corpus."""
    config = Config()
    episodes = load_archive(ARCHIVE, config, limit=limit)
    train, dev, final = episode_splits(len(episodes), config.seed)
    side = {}
    for name, index in (("train", train), ("dev", dev), ("final", final)):
        for slot in index.tolist():
            side[slot] = name
    # `eda/corpus.py` walks archive TRAIN in PERMUTATION order, not sorted slot order. Episode
    # identity is by slot either way, so the store is written in slot order and the permutation
    # survives in the split assignment itself.
    labelled = [replace(e, split=side[slot], episode_id=f"expert_v1:{slot:05d}",
                        bc_eligible=True, uniform_eligible=True,
                        terminal_cause="death" if bool(e.terminated.any()) else "timeout")
                for slot, e in enumerate(episodes)]

    out.mkdir(parents=True, exist_ok=True)
    shards, started = [], time.time()
    for position in range(0, len(labelled), per_shard):
        chunk = labelled[position:position + per_shard]
        path = out / f"shard-{position // per_shard:04d}.pt"
        record = json.loads((out / f"{path.stem}.json").read_text()) if (out / f"{path.stem}.json").exists() else None
        if record is None or not path.exists():
            record = save_episode_shard(path, chunk)
            (out / f"{path.stem}.json").write_text(json.dumps(record) + "\n")
        shards.append(record)
        print(json.dumps({"stage": "shard", "index": len(shards), "episodes": record["episodes"],
                          "seconds": round(time.time() - started, 1)}), flush=True)

    counts = {name: sum(1 for e in labelled if e.split == name) for name in ("train", "dev", "final")}
    manifest = {"format": STORE_FORMAT, "kind": "d4mj_expert_archive_v1", "complete": True,
                "episodes": len(labelled), "shards": shards,
                "transitions": sum(len(e) for e in labelled),
                "terminal_episodes": sum(bool(e.terminated.any()) for e in labelled),
                "split_episode_counts": counts,
                "split_rule": "episode_splits(len(archive), Config().seed) -- eda/corpus.py's rule, unchanged",
                "bc_eligible": True, "uniform_eligible": True,
                "source": str(ARCHIVE), "source_sha256_note": "8.6 GB; digest omitted deliberately",
                "collector_training_access": "unknown"}
    atomic_manifest(out / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/craftax_expert_store_v1")
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence")
    parser.add_argument("--per-shard", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verify", action="store_true",
                        help="load the merged corpus through load_joint_corpus and audit it")
    args = parser.parse_args(argv)
    args.evidence.mkdir(parents=True, exist_ok=True)

    manifest = merge(args.out, args.per_shard, args.limit)
    print(json.dumps({"stage": "archive_store", "episodes": manifest["episodes"],
                      "transitions": manifest["transitions"],
                      "splits": manifest["split_episode_counts"]}), flush=True)

    report = {"schema": "d4mj_m4_corpus_v1", "archive_store": manifest,
              "sources": [str(args.out), str(SUPPORT)]}
    if args.verify:
        from d4mj.config import load_recipe
        from d4mj.data import load_joint_corpus
        recipe = load_recipe(ROOT / "d4mj/recipes/lewm_mamba_raw.json")
        episodes, contract = load_joint_corpus([args.out, SUPPORT], recipe)
        audit = contract["audit"]
        rows = audit["episodes"]
        bc = [r for r in rows if r["bc"]]
        train_bc = [r for r in bc if r["split"] == "train"]
        report["contract"] = {k: v for k, v in contract.items() if k != "audit"}
        report["audit"] = {"episodes": len(rows), "transitions": audit["transitions"],
                           "split_counts": audit["split_counts"],
                           "split_digest": audit["split_digest"],
                           "bc_eligible_episodes": len(bc),
                           "bc_eligible_train_episodes": len(train_bc),
                           "bc_eligible_train_transitions": sum(r["steps"] for r in train_bc),
                           "uniform_train_episodes": sum(1 for r in rows if r["split"] == "train" and r["uniform"])}
        print(json.dumps({"stage": "merged", **report["audit"]}), flush=True)
    (args.evidence / "corpus.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "corpus_complete"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
