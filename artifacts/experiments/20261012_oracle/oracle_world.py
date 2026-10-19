"""10-12 oracle-innovation test (NOTEBOOK "10-12 ORACLE-INNOVATION TEST", predeclared): training.
E22's long recipe (dworld.train_a with long=True): the canonical L16 corrt / fmamba TWorld continued from --init, tworld's rawlong
sampler (TERMINAL_SHARE end-aligned windows, batch order seed 11), --windows L16 windows per update, the teacher loss of
tworld.rollout_losses with --recon, the phase optimizer and schedule. Rows restricted to the support-v2 rows with replayed
layouts (build.py layouts) in EVERY arm.
  --arm control   no conditioning (world.delta = None)
  --arm oracle    world.delta[:, t, tile] = Linear(4 -> 192)(true mob layout of frame t+1 at that tile), t = 0..14 (zeros at t = 15
                  and on the 18 HUD tiles); the Linear is zero-initialized, so update 0 is exactly the parent world
Logged every 500 updates: the teacher loss and, for the oracle arm, the same batch with the oracle zeroed (before the step).
Usage: oracle_world.py --arm control|oracle --recon l1|deltairis --init <world.pt> [--updates 3000] [--windows 16]
       -> artifacts/eda/oracle_v1/worlds/<name>.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260927_levers"))
import tworld as TW  # noqa: E402
S = TW.S
OUT = ROOT / "artifacts/eda/oracle_v1"


class OracleCond(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, S.D)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, nxt_layout, frames):
        """nxt_layout [B, T', 63, 4] (layout of frame t+1 for input positions t = 0..T'-1) -> delta [B, frames, 81, D]"""
        b, tp = nxt_layout.shape[:2]
        d = self.proj(nxt_layout.float())
        out = d.new_zeros(b, frames, 81, S.D)
        out[:, :tp, :63] = d
        return out


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=("control", "oracle"))
    p.add_argument("--recon", required=True, choices=("l1", "deltairis"))
    p.add_argument("--init", required=True)
    p.add_argument("--updates", type=int, default=3000)
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    name = f"{a.arm}_{a.recon}_from_{Path(a.init).stem}_u{a.updates}_b{a.windows}"
    (OUT / "worlds").mkdir(parents=True, exist_ok=True)
    log_path = ROOT / f"artifacts/eda/levers_logs/oracle_{a.arm}_{a.recon}.log"

    def log(**kw):
        print(json.dumps(kw), flush=True)
    device = torch.device("cuda")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(TW.POOLS["rawlong"] / "labels.pt", weights_only=False, mmap=True)
    n = len(pool["terminal"])
    ltok = np.memmap(TW.POOLS["rawlong"] / "tokens.f16", dtype=np.float16, mode="r", shape=(n, 64, 81, 192))
    lay = np.load(OUT / "layouts.npy", mmap_mode="r")
    valid = torch.from_numpy(np.load(OUT / "layouts_valid.npy"))
    main_rows = torch.where(~pool["terminal"] & valid)[0]
    term_rows = torch.where(pool["terminal"] & valid)[0]
    log(stage="rows", main=len(main_rows), terminal=len(term_rows), name=name)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
        world = TW.TWorld("corrt", None, "fmamba", "all", 0, False, 16)
        oc = OracleCond()
    w0 = torch.load(a.init, map_location="cpu", weights_only=False)["world"]
    if w0["time"].shape[0] != world.time.shape[0]:
        w0["time"] = w0["time"][torch.arange(world.time.shape[0]) % w0["time"].shape[0]]
    world.load_state_dict(w0)
    world, oc = world.to(device), oc.to(device)
    opt = phase_optimizer([world, oc] if a.arm == "oracle" else [world], config)
    params = [q for g in opt.param_groups for q in g["params"]]
    order = torch.Generator().manual_seed(11)
    history, started = [], time.time()
    for update in range(a.updates):
        term = torch.rand(a.windows, generator=order) < TW.TERMINAL_SHARE
        r = torch.where(term, term_rows[torch.randint(len(term_rows), (a.windows,), generator=order)],
                        main_rows[torch.randint(len(main_rows), (a.windows,), generator=order)])
        t0 = torch.where(term, 64 - 16, torch.randint(0, 49, (a.windows,), generator=order))
        s = torch.stack([torch.from_numpy(np.array(ltok[int(i), int(j):int(j) + 16])) for i, j in zip(r, t0)]).to(device).float()
        act = torch.stack([pool["actions"][int(i), int(j):int(j) + 15] for i, j in zip(r, t0)]).to(device)
        nl = torch.stack([torch.from_numpy(np.array(lay[int(i), int(j) + 1:int(j) + 16])) for i, j in zip(r, t0)]).to(device)
        with autocast_context(config):
            world.delta = oc(nl, 16) if a.arm == "oracle" else None
            teacher = TW.rollout_losses(world, s, act, "teacher", recon=a.recon)
            world.delta = None
        no_oracle = None
        if a.arm == "oracle" and (update + 1) % 500 == 0:
            with torch.no_grad(), autocast_context(config):
                no_oracle = float(TW.rollout_losses(world, s, act, "teacher", recon=a.recon))
        norm = optimizer_step(opt, teacher, params, learning_rate=_phase_lr(config, update), grad_clip=config.agent.grad_clip,
                              strict=True, zero_grad=True)
        if (update + 1) % 500 == 0:
            row = {"update": update + 1, "teacher": float(teacher), "teacher_no_oracle": no_oracle, "gradient_norm": float(norm),
                   "oracle_weight_norm": float(oc.proj.weight.norm()), "seconds": round(time.time() - started, 1),
                   "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            history.append(row)
            log(stage="train", **row)
    torch.save({"name": name, "args": vars(a), "world": world.state_dict(), "oracle": oc.state_dict(), "history": history},
               OUT / "worlds" / f"{name}.pt")
    log(stage="done", path=str(OUT / "worlds" / f"{name}.pt"))


if __name__ == "__main__":
    main()
