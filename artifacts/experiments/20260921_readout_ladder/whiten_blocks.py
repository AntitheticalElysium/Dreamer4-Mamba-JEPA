"""Post hoc: does the whitened world's TRAINED-head advantage hold on the three blocks already read?

whiten.py (sealed 58k) REPORTED: the whitened world's own trained continuation head 0.725 within root
(zombie 0.626), +0.072* / +0.127* over U's. Before a sealed replication, the cheap check: the same two
saved worlds' own heads (no fitting of any kind) on the 55k, 56k and 57k blocks, all read before.
Also U seed 2 for reference. Per block: trained W, trained U_s1, trained U_s2, DOWN (the FIT prior),
paired W - U_s1 and W - DOWN, overall and on zombie roots.

Reading (committed before the run): W - U_s1 > 0 on zombie roots in all three blocks (point estimates)
-> consistent; otherwise -> block_dependent.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N = 17
BLOCKS = {"55k": ROOT / "artifacts/eda/observe_fresh_v6", "56k": ROOT / "artifacts/eda/observe_fresh_v7",
          "57k": ROOT / "artifacts/eda/observe_fresh_v8"}
WORLDS = {"W": ("W", ROOT / "artifacts/eda/interface_worlds_white/W.pt"), "U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"),
          "U_s2": ("U", ROOT / "artifacts/eda/interface_worlds_v2/U.pt")}


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import strata
    from ladder import paired
    from observability import expected_safe, load
    import interface as I
    from whiten import whitened_pool

    pool, _ = whitened_pool()
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    pf = load(fit_seeds)["fit"]["p_death1"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    encoder, config = I.load_bridge()
    bundles = {}
    for name, (arm, path) in WORLDS.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(stored["world"])
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(stored["heads"])
        h.eval()
        bundles[name] = (arm, b, h)
    result = {}
    for block, store in BLOCKS.items():
        judge, manifest, _ = judge_store(store)
        pj, seeds = judge["p_death1"], judge["seed"]
        _, opp = expected_safe(pj, pj)
        zombie = opp & strata(judge["visible"])["zombie_adjacent"]
        safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
        for name, (arm, b, h) in bundles.items():
            out = I.branches(b, h, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device)
            safe[name] = expected_safe(out["p_dead"], pj)[0]
        test = lambda a, c, m: paired(safe[a][m], safe[c][m], seeds[m], draws=1000, seed=20261023)
        result[block] = {"manifest": manifest, "opportunity": int(opp.sum()), "zombie": int(zombie.sum()),
                         "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()},
                         "contrasts": {f"W_vs_{c}": {"overall": test("W", c, opp), "zombie": test("W", c, zombie)} for c in ("U_s1", "U_s2", "DOWN")}}
        log(block=block, **{k: round(v["overall"], 3) for k, v in result[block]["expected_safe"].items()},
            zombie={k: round(v["zombie"], 3) for k, v in result[block]["expected_safe"].items()})
    consistent = all(r["contrasts"]["W_vs_U_s1"]["zombie"]["difference"] > 0 for r in result.values())
    evidence = {"schema": "d4mj_whiten_blocks_v1", "status": "POST HOC: 55k / 56k / 57k all read before",
                "script_sha256": _sha256(Path(__file__)), "reading": "consistent" if consistent else "block_dependent", "result": result}
    (HERE / "evidence/whiten_blocks.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="whiten_blocks_complete", reading=evidence["reading"])


if __name__ == "__main__":
    main()
