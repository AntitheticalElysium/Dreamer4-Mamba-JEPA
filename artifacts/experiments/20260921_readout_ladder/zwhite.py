"""The missing cell: isotropy WITHOUT the mob. Eigen-whitened canonical z (ZW).

eigen_spectra.py: canonical SIGReg z is not isotropic (eigen spread 6,220x, effective rank 44/192) despite
equal per-coordinate variances; the whitened patch state w is (1.1x, 191.9/192) and is the only state whose
trained system works. The factorial so far:
             mob in the state      isotropic     own trained head on zombie roots
  Z (z)            no                  no             at or below DOWN
  U (u)            yes                 no             at or below DOWN
  W (w)            yes                 yes            above DOWN in 7/7 blocks
  ZW (this)        no                  yes            ?
ZW separates the isotropy effect from the information effect. State: z PCA-whitened on the TRAIN pool's
main-window frames, zw = (z - mean) V diag(lambda)^(-1/2) (all 192 components; z's smallest eigenvalue is
1/6,220 of its largest, so every direction is rescaled, as for w). interface.py's recipe unchanged, seed-1 seeds.

Scored POST HOC on the four blocks already read (55k-58k), each world's OWN continuation head, nothing
fitted: ZW, Z_s1, U_s1, W_s1, DOWN.
Readings (committed before training):
  ZW - Z_s1 on zombie roots > 0 in >= 3 of 4 blocks            -> isotropy_helps_without_mob
  W_s1 - ZW on zombie roots > 0 in >= 3 of 4 blocks            -> mob_information_needed
Both can hold (both factors matter).
"""

import argparse
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

import interface as I  # noqa: E402

N = 17
OUT = ROOT / "artifacts/eda/interface_worlds_zwhite"
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}


def zwhitened_pool():
    from whiten import whitened_pool
    pool, _ = whitened_pool()                     # installs W as well (needed to score W_s1)
    pool = dict(pool)
    z = pool["z"][~pool["terminal"]].reshape(-1, 192).double()
    mean = z.mean(0)
    values, vectors = torch.linalg.eigh(torch.cov((z - mean).T))
    T = (vectors / values.clamp_min(1e-12).sqrt()).float()           # 192 x 192
    m = mean.float()
    pool["zw"] = (pool["z"] - m) @ T
    var = pool["zw"][~pool["terminal"]][:, 1:].reshape(-1, 192).var(0)
    lam = var.rsqrt()
    pool["weights"] = dict(pool["weights"]) | {"ZW": lam / lam.mean()}
    I.KEY["ZW"] = "zw"
    base = I.state_of

    def state_of(arm, pca, zz, grid):
        return (base("Z", pca, zz, grid) - m) @ T if arm == "ZW" else base(arm, pca, zz, grid)
    I.state_of = state_of
    return pool, float(values.max() / values.min())


def train(device, log):
    pool, spread = zwhitened_pool()
    world, heads, history, counts = I.train("ZW", pool, device, log)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": "ZW", "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                "depth_counts": counts, "z_eigen_spread": spread, "script_sha256": _sha256(HERE / "interface.py"),
                "zwhite_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, OUT / "ZW.pt")
    log(status="train_complete", arm="ZW", z_eigen_spread=spread)


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import strata
    from ladder import paired
    from observability import expected_safe, load
    pool, _ = zwhitened_pool()
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    pf = load(fit_seeds)["fit"]["p_death1"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    encoder, config = I.load_bridge()
    worlds = {"ZW": ("ZW", OUT / "ZW.pt"), "Z_s1": ("Z", ROOT / "artifacts/eda/interface_worlds_v1/Z.pt"),
              "U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt"), "W_s1": ("W", ROOT / "artifacts/eda/interface_worlds_white/W.pt")}
    loaded = {}
    for name, (arm, path) in worlds.items():
        st = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(st["world"])
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(st["heads"])
        h.eval()
        loaded[name] = (arm, b, h)
    result = {}
    for block, store in BLOCKS.items():
        judge, manifest, _ = judge_store(ROOT / "artifacts/eda" / store)
        pj, seeds = judge["p_death1"], judge["seed"]
        _, opp = expected_safe(pj, pj)
        zombie = opp & strata(judge["visible"])["zombie_adjacent"]
        safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
        for name, (arm, b, h) in loaded.items():
            safe[name] = expected_safe(I.branches(b, h, pool["pca"], arm, encoder, judge["frames"], judge["actions"], device)["p_dead"], pj)[0]
        test = lambda a, c, m: paired(safe[a][m], safe[c][m], seeds[m], draws=1000, seed=20261029)
        result[block] = {"manifest": manifest,
                         "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()},
                         "contrasts": {f"{a}_vs_{c}": {"overall": test(a, c, opp), "zombie": test(a, c, zombie)}
                                       for a, c in (("ZW", "Z_s1"), ("W_s1", "ZW"), ("ZW", "DOWN"), ("U_s1", "Z_s1"))}}
        log(block=block, **{k: round(v["zombie"], 3) for k, v in result[block]["expected_safe"].items()})
    iso = sum(r["contrasts"]["ZW_vs_Z_s1"]["zombie"]["difference"] > 0 for r in result.values())
    mob = sum(r["contrasts"]["W_s1_vs_ZW"]["zombie"]["difference"] > 0 for r in result.values())
    readings = {"isotropy_helps_without_mob": iso >= 3, "mob_information_needed": mob >= 3}
    evidence = {"schema": "d4mj_zwhite_v1", "status": "POST HOC on four blocks read before", "script_sha256": _sha256(Path(__file__)),
                "readings": readings, "result": result}
    (HERE / "evidence/zwhite.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="zwhite_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "smoke"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    device = torch.device("cuda")
    if args.command == "smoke":
        pool, spread = zwhitened_pool()
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 20, 20
        I.train("ZW", pool, device, log)
        log(status="smoke_complete", z_eigen_spread=spread,
            zw_var=[round(float(v), 3) for v in pool["zw"][~pool["terminal"]].reshape(-1, 192).var(0)[[0, 96, 191]]])
    else:
        (train if args.command == "train" else score)(device, log)


if __name__ == "__main__":
    raise SystemExit(main())
