"""Native-head score of the two crossed training arms against immutable U/W refs.

Post-hoc on already inspected 55k-58k roots. The comparison isolates coordinate
scaling from the effective per-component dynamics metric within the existing
single-seed recipe; it is not a fresh gate or a replication across world seeds.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path[:0] = [str(ROOT), str(LADDER)]
import interface as I
from whiten import whitened_pool
from boundary import judge_store
from frozen_ladder import strata
from ladder import paired
from d4mj.agent import Heads
from d4mj.data import _sha256

ARMS = {"U": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"),
        "U_Wmetric": ("U", HERE / "U_Wmetric.pt"),
        "W_Umetric": ("W", HERE / "W_Umetric.pt"),
        "W": ("W", ROOT / "artifacts/eda/interface_worlds_white/W.pt")}
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7",
          "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}


def safe(p, death):
    a = p.argmin(1)
    return 1 - death.gather(1, a[:, None]).squeeze(1), a


def main():
    torch.set_num_threads(6)
    device = torch.device("cuda")
    pool, _ = whitened_pool()
    encoder, config = I.load_bridge()
    loaded = {}
    for name, (state, path) in ARMS.items():
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert data["pool_sha256"] == _sha256(I.POOL / "pool.pt")
        bundle = I.world_bundle(config, encoder, device)
        bundle.world.load_state_dict(data["world"])
        bundle.world.eval()
        heads = Heads(config).to(device)
        heads.load_state_dict(data["heads"])
        heads.eval()
        loaded[name] = (state, bundle, heads, _sha256(path))
    result, scores = {}, {}
    for block, store in BLOCKS.items():
        path = ROOT / "artifacts/eda" / store
        judge, manifest, _ = judge_store(path)
        pj, seeds = judge["p_death1"], judge["seed"]
        zombie = strata(judge["visible"])["zombie_adjacent"]
        ps, ss, choices = {}, {}, {}
        for name, (state, bundle, heads, _) in loaded.items():
            p = I.branches(bundle, heads, pool["pca"], state, encoder,
                           judge["frames"], judge["actions"], device)["p_dead"]
            ps[name] = p
            ss[name], choices[name] = safe(p, pj)
            print(block, name, "safe", round(float(ss[name][zombie].mean()), 4),
                  "sleep", round(float((choices[name][zombie] == 6).float().mean()), 4), flush=True)
        allmask = torch.ones(len(pj), dtype=torch.bool)
        contrasts = {}
        for a, b in (("U_Wmetric", "U"), ("W", "W_Umetric"),
                     ("W_Umetric", "U"), ("W", "U_Wmetric"), ("W", "U")):
            contrasts[f"{a}-{b}"] = {"zombie": paired(ss[a][zombie], ss[b][zombie], seeds[zombie], draws=1000, seed=20260926),
                                      "overall": paired(ss[a][allmask], ss[b][allmask], seeds[allmask], draws=1000, seed=20260926)}
        result[block] = {"manifest": manifest, "n": len(pj), "n_zombie": int(zombie.sum()),
                         "arms": {a: {"safe": float(ss[a].mean()), "safe_zombie": float(ss[a][zombie].mean()),
                                      "sleep_zombie": float((choices[a][zombie] == 6).float().mean()),
                                      "pdead_sleep_zombie": float(ps[a][zombie, 6].mean()),
                                      "pdead_noop_zombie": float(ps[a][zombie, 0].mean())}
                                  for a in ARMS}, "contrasts": contrasts}
        scores[block] = {"p_death1": pj, "seed": seeds, "zombie": zombie, "p_dead": ps}
    result["checkpoint_sha256"] = {name: sha for name, (*_, sha) in loaded.items()}
    (HERE / "whitening_factorial_score.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(scores, HERE / "whitening_factorial_rows.pt")


if __name__ == "__main__":
    main()
