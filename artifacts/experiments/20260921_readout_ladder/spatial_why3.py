"""Post hoc, third step: a position shortcut for death, and a copied health tile?

spatial_why2 returned `never`: one step ahead, the per-tile worlds do not generate death (TH generated
P(dead) 1e-4 against 0.64 real) or damage (4% health accuracy on generated pairs, 84% real) even on
their own TRAINING windows -- while their logged training loss on generated terminal states was near
zero. Two mechanisms would explain both, and neither needs the model to fail at optimization:

  position shortcut  In the pool a death can only occur at the window's LAST frame, time position 5:
                     windows never run past an episode's end, and terminal windows end there. The
                     continuation head reads the predictor's history, which carries a learned time
                     embedding, so "dead at position 5 after a doomed context" satisfies the loss
                     without the generated state looking dead. The eval generates position 4.
  copied health tile The per-tile L1 loss gives the health tile 1/81 of each frame's error and health
                     changes on ~8% of transitions, so predicting "no change" there is nearly free;
                     the health head's gradient did not overturn it.

On the pool's TRAIN terminal windows (6 frames, death at the last; 4,000 seeded) and the DEV split's
terminal windows (last 6 frames of each DEV terminal episode), per arm, trained P(dead) of:
  one_step        context frames 1-4 at positions 0-3, the 5th frame generated (the eval protocol)
  teacher_pos5    real frames 0-4 at positions 0-4, frame 5 predicted by the teacher pass
  rollout_pos5    the training rollout exactly: frames 0-3 real, 4 and 5 generated (as trained)
  real_pos5       the real frame 5 read with the teacher history at position 4
And, for T and TH, the health tile (token 63) of the generated frame against the real successor's and
the context frame's (the copy), on damaging transitions (dh <= -2, incl. deaths) of the same windows
plus DEV damage transitions, and on ordinary ones as a control.

Readings (committed before the run; post hoc):
  position_shortcut   teacher_pos5 or rollout_pos5 mean P(dead) >= 0.5 while one_step <= 0.1, TH, train
  health_tile_copied  generated health token nearer the copy than the real successor's on >= 80% of
                      damaging transitions, TH, train
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

from spatial import ANCHOR, ARMS, DATASET, POOL, TOKENS, WORLDS, World, bridge, encode, rollout  # noqa: E402

CAP, SEED, HEALTH_TILE = 4000, 20260928, 63


def windows(encoder, config, device, gen, log):
    """6-frame terminal windows: TRAIN from the pool, DEV from the corpus."""
    from d4mj.data import load_joint_corpus
    from exposure import health_change
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    term = torch.where(pool["terminal"])[0]
    term = term[torch.randperm(len(term), generator=gen)[:CAP]]
    train = {"z": pool["z"][term], "tokens": pool["tokens"][term], "actions": pool["actions"][term],
             "dh": pool["dh"][term]}
    record = json.loads(DATASET.read_text())
    episodes, contract = load_joint_corpus(record["paths"], config)
    if contract != record["contract"]:
        raise SystemExit("the M4 corpus differs from its dataset contract")
    dev = [e for e in episodes if e.split == "dev" and len(e) + 1 >= 6 and bool(e.terminated[-1])]
    z, tokens = encode(encoder, np.stack([np.asarray(e.observations[len(e) - 5:len(e) + 1]) for e in dev]), device)
    dev_windows = {"z": z, "tokens": tokens,
                   "actions": torch.stack([torch.as_tensor(np.asarray(e.actions_taken[len(e) - 5:])).long() for e in dev]),
                   "dh": torch.stack([health_change(np.asarray(e.rewards[len(e) - 5:], dtype=np.float64)) for e in dev])}
    log(stage="windows", train=len(term), dev=len(dev))
    return {"train": train, "dev": dev_windows}


@torch.no_grad()
def probe(world, heads, config, w, device, state, batch=32):
    from d4mj.train import autocast_context
    dead = lambda agent: (1 - torch.sigmoid(heads(agent[:, None])["continuation"][:, 0, 0].float())).cpu()
    out = {k: [] for k in ("one_step", "teacher_pos5", "rollout_pos5", "real_pos5", "tile")}
    for i in range(0, len(w["z"]), batch):
        s = (w["z"][i:i + batch, :, None] if state == "z" else w["tokens"][i:i + batch].float()).to(device)
        a = w["actions"][i:i + batch].to(device)
        with autocast_context(config):
            p1, h1 = world(s[:, 1:5], a[:, 1:5])                              # context 1-4 at positions 0-3
            out["one_step"].append(dead(world.agent(p1[:, 3], h1[:, 3])))
            pt, ht = world(s[:, :5], a[:, :5])                                # real 0-4 at positions 0-4
            out["teacher_pos5"].append(dead(world.agent(pt[:, 4], ht[:, 4])))
            out["real_pos5"].append(dead(world.agent(s[:, 5], ht[:, 4])))
            _, generated, _, generated_history = rollout(world, s, a)         # exactly as trained
            out["rollout_pos5"].append(dead(world.agent(generated[:, 1], generated_history[:, 1])))
            if state == "tokens":
                # one-step health tile at every transition t -> t+1 with 4 real context frames (t = 3, 4)
                for t in (3, 4):
                    pp, _ = world(s[:, t - 3:t + 1], a[:, t - 3:t + 1])
                    g, r, c = pp[:, 3, HEALTH_TILE], s[:, t + 1, HEALTH_TILE], s[:, t, HEALTH_TILE]
                    out["tile"].append(torch.stack([(g - r).norm(dim=-1), (g - c).norm(dim=-1), (r - c).norm(dim=-1),
                                                    w["dh"][i:i + batch, t].to(device).float()], -1).cpu())
    return {k: torch.cat(v) for k, v in out.items() if v}


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.agent import Heads
    encoder, config = bridge()
    data = windows(encoder, config, device, torch.Generator().manual_seed(SEED), log)
    del encoder
    torch.cuda.empty_cache()
    result = {}
    for arm, (state, health) in ARMS.items():
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world, heads = World(1 if state == "z" else TOKENS, health).to(device), Heads(config).to(device)
        world.load_state_dict(stored["world"])
        heads.load_state_dict(stored["heads"])
        world.eval(), heads.eval()
        result[arm] = {}
        for name, w in data.items():
            r = probe(world, heads, config, w, device, state)
            block = {k: float(r[k].mean()) for k in ("one_step", "teacher_pos5", "rollout_pos5", "real_pos5")}
            if "tile" in r:
                d_real, d_copy, change, dh = r["tile"].unbind(-1)
                for label, mask in (("damaging", dh <= -2), ("ordinary", dh == 0)):
                    block[f"tile_{label}"] = {"n": int(mask.sum()),
                                              "nearer_copy_than_real": float((d_copy[mask] < d_real[mask]).float().mean()),
                                              "generated_to_real": float(d_real[mask].mean()),
                                              "generated_to_copy": float(d_copy[mask].mean()),
                                              "real_change": float(change[mask].mean())}
            result[arm][name] = block
        log(arm=arm, **{f"{n}_{k}": (round(v, 4) if isinstance(v, float) else v["nearer_copy_than_real"])
                        for n, b in result[arm].items() for k, v in b.items()})
    th = result["TH"]["train"]
    readings = {"position_shortcut": bool(max(th["teacher_pos5"], th["rollout_pos5"]) >= 0.5 and th["one_step"] <= 0.1),
                "health_tile_copied": bool(th["tile_damaging"]["nearer_copy_than_real"] >= 0.8)}
    evidence = {"schema": "d4mj_spatial_why3_v1", "status": "POST HOC", "script_sha256": _sha256(Path(__file__)),
                "readings_TH": readings, "result": result}
    (HERE / "evidence/spatial_why3.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why3_complete", **readings)


if __name__ == "__main__":
    main()
