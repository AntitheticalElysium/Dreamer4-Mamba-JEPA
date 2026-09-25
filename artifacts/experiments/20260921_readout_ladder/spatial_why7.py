"""Post hoc, seventh step: does a continuation head trained on FACTUAL generated states, at the
evaluation position and with no terminal-position cue, choose well among COUNTERFACTUAL actions?

spatial_why6: T's generated per-tile state carries the within-root decision as well as the root does
(fresh ranking probe 0.724 vs root tokens 0.724, exploratory roots). So the 2x2's trained heads, not
the generated state, failed -- via the position shortcut (spatial_why3) and heads fitted to the real
states' large health change (spatial_why5). The repair that diagnosis implies is a supervision layout,
not a new world. But a ranking probe fitted on fork roots is not how heads are trained in imagination:
there they learn from FACTUAL transitions (one logged action per state) and are then asked about
actions never taken. This tests that directly.

Frozen worlds Z and T; nothing in them changes. A fresh continuation head is fitted with balanced BCE
on GENERATED one-step successors from the pool's own TRAIN windows, always generated at time
position 4 from 4 real context frames (the evaluation protocol): every main window's transition 3->4
(alive; damage included) and every terminal window's last transition (dead). No fork data, no real
successor state, no position cue (every example sits at the same position). Two readout inputs:
  state     the generated successor alone (T: 81 tiles, attention probe; Z: MLP)
  history   the generated successor + the predictor's history, per token (H2's readout input)
Judged within root on the exploratory observability opportunity roots (judge split): all 17 actions
advanced once, risk = P(dead), expected safe on 32-key P; three probe seeds, 3,000 updates of 256.

Readings (committed before the run; post hoc):
  T_state >= root_tokens - 0.05 (0.674)                -> heads_fixable_by_layout
  T_state <= prior + 0.05 (0.678)                      -> factual_training_does_not_transfer
  otherwise                                            -> partial
(root_tokens 0.724 and prior 0.628 on these roots, spatial_why6.)
"""

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
from d4mj.lewm_diagnostics import FORK_STORE

from confirm import seeds_for  # noqa: E402
from observability import expected_safe, load  # noqa: E402
from spatial import D, N, POOL, TOKENS, WIDTH, WORLDS, World, bridge, encode  # noqa: E402
from spatial_why4 import Probe  # noqa: E402

SEED, STEPS = 20260930, 3000


@torch.no_grad()
def factual(world, config, device, state, batch=64):
    """Generated successors at position 4 for main transitions 3->4 and terminal windows' last one."""
    from d4mj.train import autocast_context
    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    main, term = torch.where(~pool["terminal"])[0], torch.where(pool["terminal"])[0]
    jobs = [(main, 0), (term, 1)]                                   # (rows, first context frame)
    xs, hs, ys = [], [], []
    for rows, start in jobs:
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            src = pool["z"][r][:, start:start + 4, None] if state == "z" else pool["tokens"][r][:, start:start + 4]
            s = src.float().to(device)
            a = pool["actions"][r][:, start:start + 4].to(device)
            with autocast_context(config):
                predicted, history = world(s, a)
            xs.append(predicted[:, 3].float().half().cpu())
            hs.append(history[:, 3].float().half().cpu())
            ys.append((~pool["alive"][r][:, start + 4]).float())
    return torch.cat(xs), torch.cat(hs), torch.cat(ys)


@torch.no_grad()
def branches(world, config, encoder, frames, actions, device, state, batch=16):
    from d4mj.train import autocast_context
    xs, hs = [], []
    for i in range(0, len(frames), batch):
        z, tokens = encode(encoder, frames[i:i + batch, -4:], device)
        n = len(z)
        s = (z[:, :, None] if state == "z" else tokens.float()).to(device).repeat_interleave(N, 0)
        past = actions[i:i + batch, -3:].argmax(-1)
        a = torch.cat([past.repeat_interleave(N, 0), torch.arange(N).repeat(n)[:, None]], 1).to(device)
        with autocast_context(config):
            predicted, history = world(s, a)
        xs.append(predicted[:, 3].float().half().view(n, N, *predicted.shape[2:]).cpu())
        hs.append(history[:, 3].float().half().view(n, N, *history.shape[2:]).cpu())
    return torch.cat(xs), torch.cat(hs)


def head(width, tokens):
    return Probe(width, False) if tokens else nn.Sequential(nn.Flatten(), nn.Linear(width, 512), nn.ReLU(), nn.Linear(512, 1))


def fit_and_judge(x, y, xj, tokens, device, seed):
    torch.manual_seed(seed)
    model = head(x.shape[-1], tokens).to(device)
    score = (lambda v: model(v)) if tokens else (lambda v: model(v)[:, 0])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    pos = ((1 - y).sum() / y.sum()).to(device)
    for _ in range(STEPS):
        idx = torch.randint(len(x), (256,), generator=gen)
        loss = F.binary_cross_entropy_with_logits(score(x[idx].float().to(device)), y[idx].to(device), pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        flat = xj.flatten(0, 1)
        risk = torch.cat([score(flat[i:i + 1024].float().to(device)).cpu() for i in range(0, len(flat), 1024)])
    return risk.view(xj.shape[0], N)


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    judge = load(fit_seeds)["judge"]
    recorded = json.loads((HERE / "evidence/observability.json").read_text())["identity"]
    if judge["identity"] != recorded["judge"]:
        raise SystemExit("not the observability judge roots")
    opp = judge["p_death1"].amax(1) > judge["p_death1"].amin(1)
    frames, actions, p = judge["frames"][opp], judge["actions"][opp], judge["p_death1"][opp]
    del judge
    encoder, config = bridge()
    result = {}
    for arm in ("Z", "T"):
        state = "z" if arm == "Z" else "tokens"
        stored = torch.load(WORLDS / f"{arm}.pt", map_location="cpu", weights_only=False)
        world = World(1 if state == "z" else TOKENS, False).to(device)
        world.load_state_dict(stored["world"])
        world.eval()
        x, h, y = factual(world, config, device, state)
        xj, hj = branches(world, config, encoder, frames, actions, device, state)
        del world
        torch.cuda.empty_cache()
        log(stage="factual", arm=arm, examples=len(y), dead=int(y.sum()))
        for name, (train_x, judge_x) in {"state": (x, xj), "history": (torch.cat([x, h], -1), torch.cat([xj, hj], -1))}.items():
            runs = [expected_safe(fit_and_judge(train_x, y, judge_x, state == "tokens", device, s), p)[0].mean()
                    for s in range(3)]
            result[f"{arm}_{name}"] = {"expected_safe": float(np.mean([float(v) for v in runs])),
                                       "per_seed": [float(v) for v in runs]}
            log(arm=f"{arm}_{name}", safe=round(result[f"{arm}_{name}"]["expected_safe"], 4))
    reference = json.loads((HERE / "evidence/spatial_why6.json").read_text())["expected_safe"]
    t = result["T_state"]["expected_safe"]
    reading = ("heads_fixable_by_layout" if t >= reference["root_tokens"] - 0.05 else
               "factual_training_does_not_transfer" if t <= reference["prior"] + 0.05 else "partial")
    evidence = {"schema": "d4mj_spatial_why7_v1", "status": "POST HOC, EXPLORATORY: observability judge roots",
                "script_sha256": _sha256(Path(__file__)), "opportunity_roots": int(opp.sum()),
                "reference": {k: reference[k] for k in ("prior", "root_tokens", "generated_T", "generated_Z")},
                "reading": reading, "result": result}
    (HERE / "evidence/spatial_why7.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why7_complete", reading=reading)


if __name__ == "__main__":
    main()
