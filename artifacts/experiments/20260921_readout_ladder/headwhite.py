"""Why does the whitened world's trained head choose better? Head-input conditioning, or the world?

whiten / whiten_blocks: the whitened world's own trained head beats U's on zombie roots in four of four
blocks, yet its heads' fatal-vs-safe AUC on imagined states equals U's (whiten_fidelity: 0.650 vs 0.655).
Two explanations:
  conditioning  the agent readout (Linear -> LayerNorm -> GELU over cat(latent, history)) reads U's raw PCA
                latent, whose health-bearing components are up to ~48x smaller than the scroll's; per-sample
                LayerNorm does not equalize per-feature scale, so SGD barely learns the readout weights on
                them (LeCun, Bottou, Orr & Mueller 1998, Efficient BackProp: decorrelate / equalize inputs)
  world         the whitened world's imagined states themselves carry the decision better
Discriminator: UHW = the U recipe EXACTLY (U state, U loss weights, seed-1 seeds, pool, phases) except that
the agent readout reads the latent divided by W's per-component std (a fixed layer; world input, targets
and loss unchanged). Only the readout -- which every head reads -- sees a conditioned latent.

Judged POST HOC on the four blocks already read (55k, 56k, 57k, 58k), each world's OWN continuation head,
nothing fitted: UHW, U_s1, W_s1.
Readings (committed before training):
  UHW - U_s1 > 0 on zombie roots in >= 3 of 4 blocks AND |UHW - W_s1| on zombie roots < 0.03 in >= 3 of 4
      -> head_conditioning_explains_it
  UHW - U_s1 <= 0 on zombie roots in >= 3 of 4 blocks -> world_side (whitening acts through the world)
  otherwise -> mixed
"""

import argparse
import json
import os
import sys
import time
import types
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
OUT = ROOT / "artifacts/eda/headwhite_v1"
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}


def conditioned(std):
    """A world hook: the readout sees latent / std (fixed), everything else unchanged."""
    def hook(world):
        scale = std.to(next(world.parameters()).device)

        def readout(self, z, history):
            return self.agent_readout(torch.cat(((z[:, :, 0] / scale).to(history.dtype), history), -1)).unsqueeze(2)
        world.readout = types.MethodType(readout, world)
        return world
    return hook


def train(device, log):
    from whiten import whitened_pool
    _, std = whitened_pool()
    I.WORLD_HOOK = conditioned(std)
    pool = torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True)
    world, heads, history, counts = I.train("U", pool, device, log)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": "UHW", "world": world.state_dict(), "heads": heads.state_dict(), "history": history,
                "depth_counts": counts, "std": std, "script_sha256": _sha256(HERE / "interface.py"),
                "headwhite_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((I.POOL / "pool.json").read_text())["pool_sha256"]}, OUT / "UHW.pt")
    log(status="train_complete", arm="UHW", depth_counts=counts)


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_ladder import strata
    from ladder import paired
    from observability import expected_safe, load
    from whiten import whitened_pool
    pool, std = whitened_pool()
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    pf = load(fit_seeds)["fit"]["p_death1"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    encoder, config = I.load_bridge()
    worlds = {"UHW": ("U", OUT / "UHW.pt", True), "U_s1": ("U", ROOT / "artifacts/eda/interface_worlds_v1/U.pt", False),
              "W_s1": ("W", ROOT / "artifacts/eda/interface_worlds_white/W.pt", False)}
    loaded = {}
    for name, (arm, path, hooked) in worlds.items():
        stored = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(stored["world"])
        if hooked:
            conditioned(std)(b.world)
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(stored["heads"])
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
        test = lambda a, c, m: paired(safe[a][m], safe[c][m], seeds[m], draws=1000, seed=20261027)
        result[block] = {"manifest": manifest,
                         "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()},
                         "contrasts": {f"{a}_vs_{c}": {"overall": test(a, c, opp), "zombie": test(a, c, zombie)}
                                       for a, c in (("UHW", "U_s1"), ("UHW", "W_s1"), ("W_s1", "U_s1"))}}
        log(block=block, **{k: round(v["zombie"], 3) for k, v in result[block]["expected_safe"].items()})
    zu = [r["contrasts"]["UHW_vs_U_s1"]["zombie"]["difference"] for r in result.values()]
    zw = [r["contrasts"]["UHW_vs_W_s1"]["zombie"]["difference"] for r in result.values()]
    reading = ("head_conditioning_explains_it" if sum(d > 0 for d in zu) >= 3 and sum(abs(d) < 0.03 for d in zw) >= 3 else
               "world_side" if sum(d <= 0 for d in zu) >= 3 else "mixed")
    evidence = {"schema": "d4mj_headwhite_v1", "status": "POST HOC on four blocks read before", "script_sha256": _sha256(Path(__file__)),
                "reading": reading, "result": result}
    (HERE / "evidence/headwhite.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="headwhite_complete", reading=reading)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "score", "smoke"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    device = torch.device("cuda")
    if args.command == "smoke":
        from whiten import whitened_pool
        _, std = whitened_pool()
        I.WORLD_HOOK = conditioned(std)
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 20, 20
        I.train("U", torch.load(I.POOL / "pool.pt", weights_only=False, mmap=True), device, log)
        log(status="smoke_complete")
    else:
        (train if args.command == "train" else score)(device, log)


if __name__ == "__main__":
    raise SystemExit(main())
