"""Post hoc to `interface.py`: does the generated-state probe read the action-specific change Mamba
wrote, or root context plus the explicit candidate-action token?

INTERFACE.md's world-state pass read `generated_A` = the generated successor plus a candidate-action
token. U's root + action scores 0.715 against generated 0.693, so the probe could be using preserved
root context and the token (the Direct pattern) rather than the action-conditioned change. Frozen worlds,
the same FIT-train fit / FIT-dev selection and harness (all-action ranking, 3 seeds, 3,000 updates),
judged on the SAME sealed 55k roots (already read once: this is POST HOC). Per arm:

  token      the interface.py probe, refitted identically (generated state + candidate token), scored
             intact; with the 17 generated states permuted within each root (tokens kept; 5 seeded
             permutations, per-root mean); and with every state replaced by its within-root mean
  state      a probe fitted on the generated state ALONE (no token), scored intact and permuted.
             Added because a token-trained probe may lean on the token even when the state encodes the
             consequence, so its permutation drop can under-read the state.

Readings (committed before the run), U; Z reported the same way:
  state_alone_carries_consequence  state intact beats DOWN overall AND on zombie roots, and state intact
                                   beats state permuted overall (all resolved)
  root_plus_action_shortcut        token intact - token permuted not resolved > 0, AND state intact not
                                   resolved above DOWN overall
  otherwise                        mixed
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

N, DRAWS = 17, 5
RECORD, OUT = "interface.json", "interface_shuffle.json"   # replicate.py sets them


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from interface import ARMS, POOL, SEALED, WORLDS, branches, load_bridge, world_bundle
    from ladder import paired
    from observability import expected_safe, load

    recorded = json.loads((HERE / f"evidence/{RECORD}").read_text())
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, _ = seeds_for(partition, FORK_STORE)
    fit, dev = load(fit_seeds)["fit"], dev_rows(sorted(partition["fit_dev"]["seeds"]))
    judge, manifest, _ = judge_store(SEALED)
    if manifest != recorded["judge_manifest"]:
        raise SystemExit("not the block interface.py scored")
    encoder, config = load_bridge()
    gen = {}
    for arm in ARMS:
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        bundle = world_bundle(config, encoder, device)
        bundle.world.load_state_dict(stored["world"])
        bundle.world.eval()
        heads = Heads(config).to(device).eval()
        gen[arm] = {name: branches(bundle, heads, pool["pca"], arm, encoder, d["frames"], d["actions"], device)["generated"]
                    for name, d in (("fit", fit), ("dev", dev), ("judge", judge))}
        del bundle, heads
        log(stage="generated", arm=arm)
    del encoder
    torch.cuda.empty_cache()

    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    judge_rows = torch.arange(len(pj))
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    zombie = opp & strata(judge["visible"])["zombie_adjacent"]
    g = torch.Generator().manual_seed(20261010)
    perms = [torch.stack([torch.randperm(N, generator=g) for _ in range(len(pj))]) for _ in range(DRAWS)]
    rows = torch.arange(len(pj))[:, None]
    per_seed = {}

    for arm in ARMS:
        for probe, width in (("token", None), ("state", 192)):
            xf = torch.cat([gen[arm]["fit"], gen[arm]["dev"]])[..., :width]
            xj = gen[arm]["judge"][..., :width]
            mean, scale = standardize(xf, train_rows)
            xf, xj = (xf.float() - mean) / scale, (xj.float() - mean) / scale
            variants = {"intact": [xj], "permuted": [], "mean": []}
            for perm in perms:
                x = xj.clone()
                x[..., :192] = xj[rows, perm][..., :192]
                variants["permuted"].append(x)
            x = xj.clone()
            x[..., :192] = xj[..., :192].mean(1, keepdim=True)
            variants["mean"].append(x)
            runs = {k: [] for k in variants}
            for seed in range(3):
                model, _ = probe_train("branch", xf.shape[1:], xf, p, train_rows, hold_rows, seed=seed, device=device, steps=3000)
                for k, xs in variants.items():
                    runs[k].append(torch.stack([expected_safe(scores(model, x, judge_rows, device), pj)[0] for x in xs]).mean(0))
                del model
            for k in variants:
                name = f"{probe}_{arm}_{k}"
                safe[name] = torch.stack(runs[k]).mean(0)
                per_seed[name] = [float(r[opp].mean()) for r in runs[k]]
            log(arm=arm, probe=probe, **{k: round(float(safe[f"{probe}_{arm}_{k}"][opp].mean()), 4) for k in variants})

    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261011)
    up = lambda r: r["difference"] is not None and r["difference"] > 0 and r["excludes_zero"]
    contrasts = {}
    for arm in ARMS:
        for a, b in ((f"token_{arm}_intact", f"token_{arm}_permuted"), (f"token_{arm}_intact", f"token_{arm}_mean"),
                     (f"state_{arm}_intact", f"state_{arm}_permuted"), (f"state_{arm}_intact", "DOWN"),
                     (f"state_{arm}_intact", f"token_{arm}_intact")):
            contrasts[f"{a}_vs_{b}"] = {"overall": test(a, b, opp), "zombie": test(a, b, zombie)}
    c = lambda a, b, s="overall": contrasts[f"{a}_vs_{b}"][s]
    readings = {}
    for arm in ARMS:
        if up(c(f"state_{arm}_intact", "DOWN")) and up(c(f"state_{arm}_intact", "DOWN", "zombie")) and \
                up(c(f"state_{arm}_intact", f"state_{arm}_permuted")):
            readings[arm] = "state_alone_carries_consequence"
        elif not up(c(f"token_{arm}_intact", f"token_{arm}_permuted")) and not up(c(f"state_{arm}_intact", "DOWN")):
            readings[arm] = "root_plus_action_shortcut"
        else:
            readings[arm] = "mixed"
    evidence = {"schema": "d4mj_interface_shuffle_v1", "status": "POST HOC on the sealed 55k block interface.py scored",
                "script_sha256": _sha256(Path(__file__)), "interface_script_sha256": recorded["script_sha256"],
                "judge_manifest": manifest, "roots": {"opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "readings": readings,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean())} for k, v in safe.items()},
                "per_seed": per_seed, "contrasts": contrasts}
    (HERE / f"evidence/{OUT}").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="interface_shuffle_complete", **readings)


if __name__ == "__main__":
    main()
