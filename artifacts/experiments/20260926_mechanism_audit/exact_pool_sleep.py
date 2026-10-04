"""Count sleep/death visual support in the exact interface training windows.

Uses the saved (episode_id,start) ledger; reads immutable shard stores via mmap.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "artifacts/experiments/20260926_diagnosis")]
from sleep import asleep


def main():
    torch.set_num_threads(6)
    source = torch.load(ROOT / "artifacts/eda/spatial_pool_v1/pool.pt", weights_only=False, mmap=True)
    pool = torch.load(ROOT / "artifacts/eda/interface_pool_v1/pool.pt", weights_only=False, mmap=True)
    assert torch.equal(source["terminal"], pool["terminal"])
    ids = defaultdict(list)
    for index, (episode_id, start) in enumerate(source["ids"]):
        ids[episode_id].append((index, start))
    # [terminal?][rendered asleep?][alive?]
    counts = torch.zeros(2, 2, 2, dtype=torch.long)
    last = torch.zeros_like(counts)
    visited = torch.zeros(len(source["ids"]), dtype=torch.bool)
    frames_total = 0
    for store in ("craftax_expert_store_v1", "craftax_support_v2"):
        for shard in sorted((ROOT / "artifacts" / store).glob("shard-*.pt")):
            payload = torch.load(shard, weights_only=False, mmap=True)
            for e in payload["episodes"]:
                rows = ids.get(e["episode_id"])
                if not rows:
                    continue
                obs = e["observations"]
                for i, start in rows:
                    frame = obs[start:start + 6]
                    assert len(frame) == 6
                    sleeping = asleep(frame)
                    alive = pool["alive"][i].bool()
                    group = int(pool["terminal"][i])
                    for s, a in zip(sleeping, alive):
                        counts[group, int(s), int(a)] += 1
                    last[group, int(sleeping[-1]), int(alive[-1])] += 1
                    visited[i] = True
                    frames_total += 6
    assert bool(visited.all()), f"missing {int((~visited).sum())} rows"
    result = {"windows": len(visited), "terminal_windows": int(pool["terminal"].sum()),
              "frames": frames_total, "groups": {}}
    for i, name in enumerate(("main", "terminal")):
        result["groups"][name] = {"all_positions": {"awake_dead": int(counts[i, 0, 0]),
                                                     "asleep_dead": int(counts[i, 1, 0]),
                                                     "awake_alive": int(counts[i, 0, 1]),
                                                     "asleep_alive": int(counts[i, 1, 1])},
                                  "last_position": {"awake_dead": int(last[i, 0, 0]),
                                                    "asleep_dead": int(last[i, 1, 0]),
                                                    "awake_alive": int(last[i, 0, 1]),
                                                    "asleep_alive": int(last[i, 1, 1])}}
    (HERE / "exact_pool_sleep.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
