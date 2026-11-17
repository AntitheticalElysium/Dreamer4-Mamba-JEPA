"""F6.1 denoiser fine-tune with a stochastic short temporal window (PLAN.md, "Fine-tune"; SWAX, Cabannes et al. 2025,
arXiv 2509.24552, section 4.4: per batch the short window with p = 0.5, none in the last 10% of training, evaluated at
the long window). From F6.1 checkpoint 10, the denoiser alone trains, on G1 alone (perception, belief and heads frozen),
with F6.1's own streams, carry and cache (as learnable.g1_arm): per update its temporal window is 0 (each frame sees no
past frame; memory tokens are its only temporal input) with p = 0.5 before update 1,800, else the trained 4.
Readings at 1,000 and 2,000 (window 4): the natural-onset swap, the quiet bank's hazard, the monitor's G1.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.swax_ft      (GPU)
"""
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

F61 = importlib.import_module("artifacts.experiments.20261114_p1f61.train")
SW = importlib.import_module("artifacts.experiments.20261115_f61_dissection.swap")
SL = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep")
HZ = importlib.import_module("artifacts.experiments.20261115_f61_dissection.hazard")
HERE = Path(__file__).resolve().parent
OUT = HERE / "run/swax"
CKPT = HERE.parent / "20261114_p1f61/run/ckpt_10.pt"
UPDATES, READS, ANNEAL, P_SHORT, SHORT = 2000, (1000, 2000), 1800, .5, 0


def read(path, k):
    """The three readings on a saved world (window 4)."""
    res = {"swap": None, "quiet_bank": None}
    SW.run(path, f"swax_swap_{k:02d}.json")
    res["swap"] = json.loads((HERE / f"swax_swap_{k:02d}.json").read_text())
    d = OUT / f"quiet_{k:02d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "bank.pt").write_bytes((HZ.EXP / "20261107_danger_adversarial/bank.pt").read_bytes())
    SL.HERE, SL.CKPT = d, str(path)
    SL.run()
    res["quiet_bank"] = HZ.read(d / "run.pt")
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model, cfg = SW.DS.NP.IF.load(CKPT.resolve())
    dev = next(model.parameters()).device
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(model.denoiser.parameters())
    for p in params:
        p.requires_grad_(True)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    assert model.denoiser.window == 4
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=F61.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(M.MambaCarry(c.conv.to(dev), c.ssm.to(dev)) for c in ck["carry"])
    cache = ck["cache"].to(dev)
    base = ck["state"]["update"]
    rng = np.random.default_rng(20261115)
    log, losses, shorts, t0 = [], [], [], time.time()
    res = {"recipe": {"updates": UPDATES, "anneal_from": ANNEAL, "p_short": P_SHORT, "short_window": SHORT, "lr": 1e-4},
           "checkpoint_sha256": TR.file_sha256(CKPT), "log": log, "reads": {}}
    for u in range(1, UPDATES + 1):
        short = u <= ANNEAL and rng.random() < P_SHORT
        model.denoiser.window = SHORT if short else 4
        chunk = streams.next(64)
        boot = TR.bootstrap_rows(cfg, F61.WORLD_SEED, base + u)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                   None, cfg, boot[rows], base + u, None)
            (terms["g1"] / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            losses.append((float(terms["g1"]), short))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, flush=True)
        if u % 100 == 0:
            recent = losses[-200:]
            log.append({"update": u, "g1_long": float(np.mean([l for l, s in recent if not s] or [np.nan])),
                        "g1_short": float(np.mean([l for l, s in recent if s] or [np.nan])), "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(log[-1]), flush=True)
        if u in READS:
            model.denoiser.window = 4
            model.eval()
            mon = TR.monitor(model, cfg, split["monitor"])
            path = OUT / f"ckpt_{u:04d}.pt"
            torch.save({"model": model.state_dict(), "cfg": cfg}, path)
            res["reads"][u] = {"monitor_g1": float(mon["g1"]), **read(path, u // 100)}
            r = res["reads"][u]
            print("read", u, json.dumps({"monitor_g1": r["monitor_g1"], "swap_rate": r["swap"]["rate"],
                                         "hazard": r["quiet_bank"]["hazard_by_step_relative_to_due"],
                                         "rank": r["quiet_bank"]["spearman_true_vs_generated_median"]}), flush=True)
            model.train()
            (HERE / "swax_ft.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
