"""The combined recipe: whitened patch state + full-corpus data + a dense harm head trained on imagined states.

Two findings to combine, both inside the factual contract (apart from the declared u-state deviation):
  * whiten.py / whiten_blocks.py: with the isotropic (whitened) patch state, the world's OWN trained head
    beats DOWN overall in four of four blocks and U's trained head on zombie roots in four of four.
  * factual_scale.py: the consequence is learnable from factual data only in its DENSE form (harm = death
    OR 2+ health lost) and only at full corpus scale (0.689; neither ingredient alone helps).
genscale.py showed a head trained after the fact on U's imagined states cannot use the dense label: the
imagined state does not draw damage. The TD-MPC2 / MuZero answer is to train the decision head ON the
rolled-out states WITH gradients into the dynamics, so the transition is shaped to carry the outcome.

Pool (full scale, from u_cache_train_v1, whitened with the SAME per-component std as W): 400,000 uniform
6-frame main windows over every TRAIN transition (seeded) + every TRAIN death-ending window (8,071);
rewards from the corpus. Arms, both = interface.py's recipe on this pool (seed-1 seeds, 1x budget,
alias-free bridge):
  WF     whitened, full-scale pool
  WFH    WF + a harm head (256 -> 256 -> 1 on the pooled agent readout) on observed AND imagined frames,
         class-balanced BCE, 0.5 observed / 0.5 generated, 0.8 main / 0.2 terminal, its own RMS-balanced
         loss group (interface.EXTRA_LOSS), gradients into the world
Reference: W seed 1 (the pool-trained whitened world).

Judge: a NEW block, `observe.py collect --seed-start 60000 --target-opportunity 800 --max-seeds 1500
--out artifacts/eda/observe_fresh_v11`, collected after this commit, read once. Each world's OWN heads
on its imagined successors (4 observed frames, one advance per action): the continuation head for every
arm, and WFH's harm head. actions_only (frozen_ladder, FIT-train / FIT-dev, 3 seeds); DOWN.

DECLARED READINGS (committed before collection and before training):
  primary    WFH's harm head: - DOWN overall, - actions_only overall, - DOWN on zombie roots, all
             resolved > 0 -> trained_system_passes; else trained_system_fails
  secondary  WFH harm head - W_s1 continuation head on zombie roots resolved > 0 -> combined_recipe_helps
Reported: WF - W_s1 (data scale for the world), WFH continuation - WF continuation (the harm head's effect
on the world), night / lava strata, SLEEP choices.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

import interface as I  # noqa: E402

N, W, MAIN_WINDOWS, SEED = 17, 6, 400_000, 20261025
POOL = ROOT / "artifacts/eda/harm_pool_v1"
OUT = ROOT / "artifacts/eda/harm_worlds_v1"
JUDGE = ROOT / "artifacts/eda/observe_fresh_v11"


def build(log):
    from d4mj.data import load_joint_corpus
    from factual_scale import CACHE, load_cache
    from whiten import whitened_pool
    ref, std = whitened_pool()
    U, A, dead, dh, ids, _ = load_cache()
    config = I.load_bridge()[1]
    record = json.loads(I.DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    rewards = {e.episode_id: torch.as_tensor(np.asarray(e.rewards), dtype=torch.float32) for e in episodes
               if e.episode_id in ids}
    names = sorted(ids)
    starts = torch.tensor([ids[k][1] + 1 - W + 1 for k in names]).clamp(min=0)          # valid starts per episode
    g = torch.Generator().manual_seed(SEED)
    pick = torch.multinomial(starts.double(), MAIN_WINDOWS, replacement=True, generator=g)
    windows = [(names[i], int(torch.randint(int(starts[i]), (), generator=g))) for i in pick.tolist()]
    windows += [(k, ids[k][1] + 1 - W) for k in names if bool(dead[ids[k][0] + ids[k][1] - 1])]
    n = len(windows)
    pool = {"w": torch.empty(n, W, 192), "actions": torch.empty(n, W - 1, dtype=torch.long),
            "reward_led": torch.zeros(n, W), "reward_valid": torch.zeros(n, W, dtype=torch.bool),
            "alive": torch.ones(n, W, dtype=torch.bool), "dh": torch.empty(n, W - 1, dtype=torch.long)}
    for j, (k, s) in enumerate(windows):
        off, length = ids[k]
        pool["w"][j] = U[off + s: off + s + W].float() / std
        pool["actions"][j] = A[off + s: off + s + W - 1]
        pool["dh"][j] = dh[off + s: off + s + W - 1].long()
        r = rewards[k]
        for f in range(W):
            t = s + f - 1
            if t >= 0:
                pool["reward_led"][j, f], pool["reward_valid"][j, f] = r[t], True
            if s + f == length and bool(dead[off + length - 1]):
                pool["alive"][j, f] = False
        if j % 100_000 == 0:
            log(stage="building", done=j, of=n)
    pool["terminal"] = torch.arange(n) >= MAIN_WINDOWS
    if not bool((~pool["alive"][pool["terminal"], -1]).all()) or bool((~pool["alive"][:, :-1]).any()):
        raise SystemExit("terminal layout broken")
    main = ~pool["terminal"]
    var = pool["w"][main][:, 1:].reshape(-1, 192).var(0)
    lam = var.rsqrt()
    pool["weights"] = {"W": lam / lam.mean()}
    pool["pca"] = ref["pca"]
    harm = ~pool["alive"][:, 1:] | (pool["dh"] <= -2)
    POOL.mkdir(parents=True, exist_ok=True)
    torch.save(pool, POOL / "pool.pt")
    meta = {"windows": n, "main": MAIN_WINDOWS, "terminal": n - MAIN_WINDOWS, "harm_frames": int(harm.sum()),
            "natural_terminal_in_main": int((~pool["alive"][main, -1]).sum()), "std_from": "whiten.whitened_pool",
            "cache_manifest_sha256": _sha256(CACHE / "manifest.json"), "pool_sha256": _sha256(POOL / "pool.pt")}
    (POOL / "pool.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(status="pool_complete", **meta)


class Harm(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, features):                 # [B, T, tokens, 256] -> [B, T]
        return self.net(features.mean(2).float())[..., 0]


def train(arm, device, log):
    from whiten import whitened_pool
    whitened_pool()                               # installs the W arm and its state_of in interface
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    I.KEY["W"] = "w"
    harm_frames = ~pool["alive"][:, 1:] | (pool["dh"] <= -2)                                   # frame f>=1
    labels = torch.cat([torch.zeros(len(harm_frames), 1, dtype=torch.bool), harm_frames], 1).float()
    head = None
    if arm == "WFH":
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(7 + 5)
            head = Harm().to(device)
        rate = float(labels[:, 1:].mean())
        pos = torch.tensor((1 - rate) / rate, device=device)

        def extra(parts, main, term, shift):
            total = 0.0
            for name, rows, weight in (("main", main, 0.8), ("terminal", term, 0.2)):
                _, teacher, _, gfeat, anchor, _ = parts[name]
                y = labels[rows][:, shift:].to(device)
                bce = lambda logit, target: F.binary_cross_entropy_with_logits(logit, target, pos_weight=pos)
                total = total + weight * (0.5 * bce(head(teacher.features)[:, 1:], y[:, 1:])
                                          + 0.5 * bce(head(gfeat), y[:, anchor + 1:]))
            return {"harm": total}
        I.EXTRA_MODULES, I.EXTRA_LOSS = (head,), extra
    world, heads, history, counts = I.train("W", pool, device, log)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": arm, "world": world.state_dict(), "heads": heads.state_dict(),
                "harm": None if head is None else head.state_dict(), "history": history, "depth_counts": counts,
                "script_sha256": _sha256(HERE / "interface.py"), "harmworld_sha256": _sha256(Path(__file__)),
                "pool_sha256": json.loads((POOL / "pool.json").read_text())["pool_sha256"]}, OUT / f"{arm}.pt")
    log(status="train_complete", arm=arm, depth_counts=counts)


def score(device, log):
    from d4mj.agent import Heads
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from frozen_heads import dev_rows
    from frozen_ladder import scores, standardize, strata, train as probe_train
    from ladder import paired
    from observability import expected_safe, load
    from whiten import whitened_pool

    ref, _ = whitened_pool()
    partition = json.loads((HERE / "evidence/root_partition.json").read_text())
    fit_seeds, unallocated = seeds_for(partition, FORK_STORE)
    dev_seeds = sorted(partition["fit_dev"]["seeds"])
    forbidden = set(partition["reserved_for_gate"]["seeds"]) | set(unallocated)
    fit, dev = load(fit_seeds)["fit"], dev_rows(dev_seeds)
    judge, manifest, files = judge_store(JUDGE)
    new = set(judge["seed"].unique().tolist())
    used = {int(f.stem.split("-")[1]) for k in range(1, 11) for f in (ROOT / f"artifacts/eda/observe_fresh_v{k}").glob("seed-*.pt")}
    if min(new) < 60_000 or new & used or new & (set(fit_seeds) | set(dev_seeds) | forbidden):
        raise SystemExit("judgement seeds are not a new, untouched block")
    pf, pd, pj, seeds = fit["p_death1"], dev["p_death1"], judge["p_death1"], judge["seed"]
    prior = int(pf[pf.amax(1) > pf.amin(1)].mean(0).argmin())
    safe = {"DOWN": expected_safe(-F.one_hot(torch.full((len(pj),), prior), N).float(), pj)[0]}
    _, opp = expected_safe(pj, pj)
    strat = strata(judge["visible"])
    zombie = opp & strat["zombie_adjacent"]
    p = torch.cat([pf, pd])
    train_rows, hold_rows = torch.arange(len(pf)), torch.arange(len(pf), len(p))
    actions4 = torch.cat([fit["actions"][:, -4:], dev["actions"][:, -4:]]).flatten(1)
    mean, scale = standardize(actions4, train_rows)
    runs = []
    for seed in range(3):
        model, _ = probe_train("vector", actions4.shape[1:], (actions4 - mean) / scale, p, train_rows, hold_rows,
                               seed=seed, device=device, steps=3000)
        runs.append(expected_safe(scores(model, (judge["actions"][:, -4:].flatten(1) - mean) / scale,
                                         torch.arange(len(pj)), device), pj)[0])
    safe["actions_only"] = torch.stack(runs).mean(0)
    encoder, config = I.load_bridge()
    sleep = {}
    for name, path in (("W_s1", ROOT / "artifacts/eda/interface_worlds_white/W.pt"), ("WF", OUT / "WF.pt"), ("WFH", OUT / "WFH.pt")):
        stored = torch.load(path, map_location="cpu", weights_only=False)
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(stored["world"])
        b.world.eval()
        h = Heads(config).to(device)
        h.load_state_dict(stored["heads"])
        h.eval()
        out = I.branches(b, h, ref["pca"], "W", encoder, judge["frames"], judge["actions"], device)
        safe[f"{name}_continuation"] = expected_safe(out["p_dead"], pj)[0]
        sleep[f"{name}_continuation"] = int((out["p_dead"][opp].argmin(1) == 6).sum())
        if stored.get("harm") is not None:
            head = Harm().to(device)
            head.load_state_dict(stored["harm"])
            head.eval()
            with torch.no_grad():
                risk = torch.cat([head(out["features"][j:j + 256, :, None, :256].to(device)).cpu()
                                  for j in range(0, len(pj), 256)])
            safe[f"{name}_harm"] = expected_safe(risk, pj)[0]
            sleep[f"{name}_harm"] = int((risk[opp].argmin(1) == 6).sum())
        del b, h
        log(world=name, **{k: round(float(v[opp].mean()), 4) for k, v in safe.items() if k.startswith(name)})
    test = lambda a, b, m: paired(safe[a][m], safe[b][m], seeds[m], draws=1000, seed=20261026)
    up = lambda r: r["difference"] > 0 and r["excludes_zero"]
    rules = {"WFH_harm_vs_DOWN": test("WFH_harm", "DOWN", opp), "WFH_harm_vs_actions_only": test("WFH_harm", "actions_only", opp),
             "WFH_harm_vs_DOWN_zombie": test("WFH_harm", "DOWN", zombie),
             "WFH_harm_vs_W_s1_zombie": test("WFH_harm", "W_s1_continuation", zombie)}
    readings = {"primary": ("trained_system_passes" if up(rules["WFH_harm_vs_DOWN"]) and up(rules["WFH_harm_vs_actions_only"])
                            and up(rules["WFH_harm_vs_DOWN_zombie"]) else "trained_system_fails"),
                "secondary": "combined_recipe_helps" if up(rules["WFH_harm_vs_W_s1_zombie"]) else "no_evidence"}
    reported = {f"{a}_vs_{b}": {"overall": test(a, b, opp), "zombie": test(a, b, zombie)} for a, b in (
        ("WF_continuation", "W_s1_continuation"), ("WFH_continuation", "WF_continuation"),
        ("WFH_harm", "WFH_continuation"), ("W_s1_continuation", "DOWN"), ("WF_continuation", "DOWN"))}
    evidence = {"schema": "d4mj_harmworld_v1", "status": "SEALED: new seed block, rules committed before collection",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "judge_seed_files": files,
                "roots": {"judge": len(pj), "opportunity": int(opp.sum()), "zombie": int(zombie.sum())},
                "rules": rules, "readings": readings, "reported": reported, "sleep_choices": sleep,
                "expected_safe": {k: {"overall": float(v[opp].mean()), "zombie": float(v[zombie].mean()),
                                      **{s: float(v[m & opp].mean()) for s, m in strat.items() if (m & opp).any()}}
                                  for k, v in safe.items()}}
    (HERE / "evidence/harmworld.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="harmworld_complete", **readings)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pool", "train", "score", "smoke"))
    parser.add_argument("--arm", choices=("WF", "WFH"))
    args = parser.parse_args(argv)
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    os.chdir(ROOT)
    device = torch.device("cuda")
    if args.command == "pool":
        build(log)
    elif args.command == "smoke":
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 20, 20
        global OUT
        OUT = Path("/tmp/claude-1000/-home-antithetical-EPITA-PERSO-DynamicHorizons-Mamba-JEPA/453c4e0b-57dd-4ecb-a41b-c69a33183014/scratchpad/harm_smoke")
        train(args.arm, device, log)
    elif args.command == "train":
        train(args.arm, device, log)
    else:
        score(device, log)


if __name__ == "__main__":
    raise SystemExit(main())
