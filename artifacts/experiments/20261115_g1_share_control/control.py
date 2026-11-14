"""The matched control for the learnability A5 arm (PLAN.md): A5's continuation with F6's full weighted loss, then the
held-out monitor for F6, A5 and this control, and F6's clock readings on the control's saved worlds.

    MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu python -m artifacts.experiments.20261115_g1_share_control.control [train|monitor|readings]
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
LB = importlib.import_module("artifacts.experiments.20261114_learnability.learnable")
DZ, DEV = LB.DZ, LB.DEV


def train(updates=2000, evals=(1000, 2000)):
    """learnable.g1_arm line for line, except the loss (F6's harmonised groups plus the scale term)."""
    out = HERE / "control.json"
    if out.exists():
        return
    torch.manual_seed(0)
    ck_path, model, cfg, _ = DZ.load(10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    params = list(model.belief.parameters()) + list(model.denoiser.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=DZ.F6.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(M.MambaCarry(c.conv.to(DEV), c.ssm.to(DEV)) for c in ck["carry"])
    cache = ck["cache"].to(DEV)
    res = {"arm": "control_full_loss", "log": [], "sigma": model.harmoniser.log_sigma.exp().tolist()}
    t0, losses = time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        boot = TR.bootstrap_rows(cfg, DZ.F6.WORLD_SEED, 15625 + u)
        f = min(1.0, u / 100)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * f
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                   None, cfg, boot[rows], 15625 + u, None)
            ((model.harmoniser(TR.groups(terms)) + terms["noise"]) / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            losses.append(float(terms["g1"]))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u % 250 == 0:
            res["log"].append({"update": u, "g1": float(np.mean(losses[-500:])), "sec_per_update": (time.time() - t0) / u})
            print("control", res["log"][-1], flush=True)
        if u in evals:
            run = HERE / "run"
            run.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, run / f"ckpt_{evals.index(u) + 1:02d}.pt")
    out.write_text(json.dumps(res, indent=1) + "\n")


def monitor():
    out = HERE / "monitor.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    cfg0 = torch.load(DZ.F6.HERE / "run/ckpt_10.pt", map_location="cpu", weights_only=False)["cfg"]
    episodes = D.load_split(cfg0)["monitor"]
    worlds = {"f6_ckpt10": DZ.F6.HERE / "run/ckpt_10.pt",
              "a5_g1_alone_1000": LB.HERE / "g1_run/run/ckpt_01.pt", "a5_g1_alone_2000": LB.HERE / "g1_run/run/ckpt_02.pt",
              "control_1000": HERE / "run/ckpt_01.pt", "control_2000": HERE / "run/ckpt_02.pt"}
    for name, path in worlds.items():
        if name in res:
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = M.M0(ck["cfg"]).to(DEV)
        model.load_state_dict(ck["model"])
        torch.manual_seed(0)
        m = TR.monitor(model, ck["cfg"], episodes)
        res[name] = {"sha256": TR.file_sha256(path), **{k: v for k, v in m.items() if k != "raw_token_statistics"}}
        print(name, {k: res[name][k] for k in ("g1", "g1_by_level") if k in res[name]}, flush=True)
        out.write_text(json.dumps(res, indent=1) + "\n")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    stage = sys.argv[1] if len(sys.argv) > 1 else "train"
    if stage == "train":
        train()
    elif stage == "monitor":
        monitor()
    elif stage == "readings":
        TJ = importlib.import_module("artifacts.experiments.20261111_p1f5.trajectory")
        TJ.HERE, TJ.OUT = HERE, HERE / "readings"
        TJ.OUT.mkdir(parents=True, exist_ok=True)
        for k in (1, 2):
            TJ.checkpoint(k)
