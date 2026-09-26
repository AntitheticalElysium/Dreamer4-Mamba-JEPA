"""Encode the whole M4 TRAIN corpus into U's state space once (an asset, not an experiment).

Every model in the interface line saw only the 32,647-window pool: every TRAIN death, but ~1.6% of the
transitions. The corpus holds 235,863 near-zombie transitions (staying loses 2+ health 21.9% of the
time, n=110,485; moving 7.1%, n=125,378: exposure.json m4_corpus) -- the mechanism, ~60x more often than
the pool shows it. This caches, for every TRAIN, uniform-eligible episode of the dataset contract:
  u        [len+1, 192] fp16   the frozen Raw H2 encoder's 4x4 pooled patch grid through the interface
                               pool's TRAIN-fitted PCA (interface_pool_v1, pinned by hash)
  actions  [len] int64, dead [len] bool (terminated), dh [len] int8 (health change from the reward,
           exposure.health_change, exact on 7,021 fork transitions)
Shards of 250 episodes under artifacts/eda/u_cache_train_v1, with a manifest (episode ids, hashes).
Resumable: finished shards are skipped.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

OUT = ROOT / "artifacts/eda/u_cache_train_v1"
SHARD = 250


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.data import load_joint_corpus
    from exposure import health_change
    from interface import DATASET, POOL, encode, load_bridge, project
    pool_meta = json.loads((POOL / "pool.json").read_text())
    if _sha256(POOL / "pool.pt") != pool_meta["pool_sha256"]:
        raise SystemExit("interface pool changed")
    pca = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)["pca"]
    encoder, config = load_bridge()
    record = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    train = [e for e in episodes if e.split == "train" and e.uniform_eligible]
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [train[i:i + SHARD] for i in range(0, len(train), SHARD)]
    log(stage="plan", episodes=len(train), shards=len(shards), frames=int(sum(len(e) + 1 for e in train)))
    for k, shard in enumerate(shards):
        path = OUT / f"shard-{k:04d}.pt"
        if path.exists():
            continue
        rows = []
        for e in shard:
            obs = np.asarray(e.observations)
            us = []
            for i in range(0, len(obs), 512):
                _, grid = encode(encoder, obs[i:i + 512][:, None], device, batch=64)
                us.append(project(pca, grid[:, 0]).half())
            rows.append({"id": e.episode_id, "u": torch.cat(us), "actions": torch.as_tensor(np.asarray(e.actions_taken)).long(),
                         "dead": torch.as_tensor(np.asarray(e.terminated)).bool(),
                         "dh": health_change(np.asarray(e.rewards, dtype=np.float64)).to(torch.int8)})
        torch.save(rows, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)
        log(stage="shard", k=k, of=len(shards))
    manifest = {"episodes": len(train), "shards": len(shards), "pool_sha256": pool_meta["pool_sha256"],
                "checkpoint_sha256": pool_meta["checkpoint_sha256"], "contract_sha256": record["contract"]["sha256"],
                "script_sha256": _sha256(Path(__file__)),
                "shard_sha256": {p.name: _sha256(p) for p in sorted(OUT.glob("shard-*.pt"))}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(status="ucache_complete", episodes=len(train))


if __name__ == "__main__":
    main()
