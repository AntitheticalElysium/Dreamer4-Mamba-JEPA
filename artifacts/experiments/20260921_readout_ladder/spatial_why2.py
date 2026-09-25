"""Post hoc, second step: why does the per-tile world never GENERATE the consequence at fork roots?

spatial_why.py: in T and TH the trained heads read death on the REAL successor at within-root AUC
0.998 / 0.996, and TH's health head is 98% right on real damaged pairs -- but on GENERATED
successors P(dead) is ~1e-4 on fatal branches and the health head is right on 0% of damaged pairs.
Yet in training the continuation loss on generated terminal states was near zero (paired terminal
loss, logged). So the world did generate death on its training windows. Three explanations remain:

  memorized   it reproduces deaths on the TRAIN windows it saw ~20 times each, not on held-out
              windows of the same corpus
  shifted     it generates them on held-out corpus windows too, but not at the fork roots (another
              policy's states, or a counterfactual action there)
  never       it does not generate them even on its own TRAIN windows one step ahead

Measured one step ahead with the eval protocol (4 context frames at time positions 0-3 with their 3
actions, the logged action at frame 3, the 5th frame generated), frozen worlds, nothing refit, on:
  train   the pool's own windows: terminal windows (death at the last frame) and main windows whose
          transition 3->4 or 4->5 loses 2+ health (each capped at 4,000, seeded)
  dev     the M4 corpus DEV split, never in the pool: every terminal episode's last 5 frames, and
          transitions losing 2+ health, plus ordinary (dh = 0) transitions (capped, seeded)
  fork    the sealed roots, from spatial_why.json
Per population and arm: mean trained P(dead) on generated vs real terminal successors (and the share
above 0.5); the health head's accuracy on generated vs real damaged pairs (ZH, TH); generated P(dead)
on ordinary transitions (false alarms).

Reading for TH (committed before the run; post hoc):
  generated terminal P(dead) >= 0.5 on train and < 0.2 on dev   -> memorized
  >= 0.5 on train and on dev                                     -> shifted
  < 0.5 on train                                                 -> never
  otherwise                                                      -> mixed
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

from spatial import ARMS, DATASET, POOL, TOKENS, WORLDS, World, bridge, encode  # noqa: E402

CAP, SEED = 4000, 20260927


def subsample(idx, gen):
    return idx[torch.randperm(len(idx), generator=gen)[:CAP]] if len(idx) > CAP else idx


def train_population(gen):
    """Windows of 5 frames from the pool: z, tokens, 4 actions, dh of the last transition, terminal."""
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    parts = []
    term = subsample(torch.where(pool["terminal"])[0], gen)
    parts.append(("terminal", term, 1))
    for t in (3, 4):
        main = torch.where(~pool["terminal"] & (pool["dh"][:, t] <= -2))[0]
        parts.append((f"damage_t{t}", subsample(main, gen), t - 3))
    out = {k: [] for k in ("z", "tokens", "actions", "dh", "terminal")}
    for _, rows, start in parts:
        out["z"].append(pool["z"][rows][:, start:start + 5])
        out["tokens"].append(pool["tokens"][rows][:, start:start + 5])
        out["actions"].append(pool["actions"][rows][:, start:start + 4])
        out["dh"].append(pool["dh"][rows][:, start + 3])
        out["terminal"].append(~pool["alive"][rows][:, start + 4])
    return {k: torch.cat(v) for k, v in out.items()}


def dev_population(encoder, config, device, gen, log):
    from d4mj.data import load_joint_corpus
    from exposure import health_change
    record = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    dev = [e for e in episodes if e.split == "dev" and e.uniform_eligible and len(e) + 1 >= 5]
    terminal, damage, ordinary = [], [], []
    for e in dev:
        dh = health_change(np.asarray(e.rewards, dtype=np.float64))
        if bool(e.terminated[-1]):
            terminal.append((e, len(e) - 4))
        for t in range(3, len(e) - (1 if bool(e.terminated[-1]) else 0)):
            (damage if int(dh[t]) <= -2 else ordinary).append((e, t - 3))
    pick = lambda xs: [xs[i] for i in torch.randperm(len(xs), generator=gen)[:CAP].tolist()]
    windows = terminal[:CAP] + pick(damage) + pick(ordinary)
    log(stage="dev_windows", terminal=len(terminal[:CAP]), damage=min(CAP, len(damage)), ordinary=min(CAP, len(ordinary)))
    zs, ts, acts, dhs, terms = [], [], [], [], []
    for b in range(0, len(windows), 256):
        chunk = windows[b:b + 256]
        z, tokens = encode(encoder, np.stack([np.asarray(e.observations[s:s + 5]) for e, s in chunk]), device)
        zs.append(z)
        ts.append(tokens)
        for e, s in chunk:
            acts.append(torch.as_tensor(np.asarray(e.actions_taken[s:s + 4])).long())
            dhs.append(int(health_change(np.asarray([e.rewards[s + 3]], dtype=np.float64))[0]))
            terms.append(bool(e.terminated[s + 3]))
    return {"z": torch.cat(zs), "tokens": torch.cat(ts), "actions": torch.stack(acts),
            "dh": torch.tensor(dhs), "terminal": torch.tensor(terms)}


@torch.no_grad()
def one_step(world, heads, config, population, device, state, health, batch=64):
    from d4mj.train import autocast_context
    out = {k: [] for k in ("gen", "real", "dh_gen", "dh_real")}
    for i in range(0, len(population["z"]), batch):
        s = (population["z"][i:i + batch, :, None] if state == "z"
             else population["tokens"][i:i + batch].float()).to(device)
        a = population["actions"][i:i + batch].to(device)
        with autocast_context(config):
            predicted, history = world(s[:, :4], a)
            for key, succ in (("gen", predicted[:, 3]), ("real", s[:, 4])):
                agent = world.agent(succ, history[:, 3])[:, None]
                out[key].append((1 - torch.sigmoid(heads(agent)["continuation"][:, 0, 0].float())).cpu())
                if health:
                    out[f"dh_{key}"].append(world.health(s[:, 3], succ).float().argmax(-1).cpu() - 9)
    return {k: torch.cat(v) for k, v in out.items() if v}


def summarize(r, population, health):
    term, dh = population["terminal"], population["dh"]
    damaged = (dh <= -2) & ~term
    ordinary = (dh == 0) & ~term
    block = {"n_terminal": int(term.sum()), "n_damaged": int(damaged.sum()), "n_ordinary": int(ordinary.sum())}
    for key in ("gen", "real"):
        block[f"{key}_p_dead_terminal"] = float(r[key][term].mean()) if term.any() else None
        block[f"{key}_share_dead_terminal"] = float((r[key][term] > 0.5).float().mean()) if term.any() else None
        block[f"{key}_p_dead_ordinary"] = float(r[key][ordinary].mean()) if ordinary.any() else None
        if health and damaged.any():
            block[f"{key}_health_accuracy_damaged"] = float((r[f"dh_{key}"][damaged] == dh[damaged]).float().mean())
            block[f"{key}_health_accuracy_terminal"] = float((r[f"dh_{key}"][term] == dh[term]).float().mean()) if term.any() else None
    return block


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    gen = torch.Generator().manual_seed(SEED)
    encoder, config = bridge()
    populations = {"train": train_population(gen), "dev": dev_population(encoder, config, device, gen, log)}
    del encoder
    torch.cuda.empty_cache()
    log(stage="populations", **{k: len(v["z"]) for k, v in populations.items()})
    result = {}
    for arm, (state, health) in ARMS.items():
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world, heads = World(1 if state == "z" else TOKENS, health).to(device), Heads(config).to(device)
        world.load_state_dict(stored["world"])
        heads.load_state_dict(stored["heads"])
        world.eval(), heads.eval()
        result[arm] = {name: summarize(one_step(world, heads, config, p, device, state, health), p, health)
                       for name, p in populations.items()}
        log(arm=arm, **{f"{n}_{k}": (round(v, 4) if isinstance(v, float) else v)
                        for n, b in result[arm].items() for k, v in b.items() if k.startswith(("gen_p_dead_t", "real_p_dead_t", "gen_health_accuracy_d", "real_health_accuracy_d"))})
    th = result["TH"]
    train_dead, dev_dead = th["train"]["gen_p_dead_terminal"], th["dev"]["gen_p_dead_terminal"]
    reading = ("memorized" if train_dead >= 0.5 and dev_dead < 0.2 else "shifted" if train_dead >= 0.5 and dev_dead >= 0.5
               else "never" if train_dead < 0.5 else "mixed")
    fork = json.loads((HERE / "evidence/spatial_why.json").read_text())["result"]
    evidence = {"schema": "d4mj_spatial_why2_v1", "status": "POST HOC", "script_sha256": _sha256(Path(__file__)),
                "reading_TH": reading, "result": result,
                "fork_from_spatial_why": {a: {k: v for k, v in b.items() if "p_dead" in k or "damage" in k} for a, b in fork.items()}}
    (HERE / "evidence/spatial_why2.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why2_complete", reading_TH=reading)


if __name__ == "__main__":
    main()
