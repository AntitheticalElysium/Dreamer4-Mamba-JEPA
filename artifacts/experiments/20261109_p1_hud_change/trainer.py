"""P1-HUD-Change trainer: identical to saccade.train.p1 with optional controller_cfg hook;
test_contract.py checks its AST against TR.p1."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from saccade import chooser as C, data as D, model as M, train as TR


def p1(cfg, world_seed, out_dir: Path, updates=None, steps=None, n_episodes=None, checkpoints=True, log_every=25, controller_cfg=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(world_seed)
    split = D.load_split(cfg)
    eps = split["train"][:n_episodes] if n_episodes else split["train"]
    targets = np.load(D.targets_dir(cfg) / "change_targets.npy", mmap_mode="r")
    offsets = D.frame_offsets(split["train"])
    D.validate_change_targets(targets, split["train"], D.targets_dir(cfg))
    provenance = TR.run_provenance(cfg)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    assert max(d["accumulate_at_64"], d["accumulate_at_256"]) <= d["streams"], "a microbatch without streams"
    streams = D.Streams(eps, d["streams"], P, seed=world_seed, targets_offsets=offsets)
    model = M.M0(cfg).to(dev)
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    TR.set_reward_values(model, split["train"])
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    controller = C.Controller(model, cfg if controller_cfg is None else dict(cfg, fixation=dict(chooser=controller_cfg)), out_dir, dev)
    W = cfg["belief"]["width"]
    st = {"frames": 0, "update": 0, "next_ckpt": 1}
    carry = model.belief.initial(d["streams"], dev)
    cache = torch.zeros(d["streams"], P + 1, W, device=dev)
    latest = out_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, weights_only=False, map_location=dev)
        TR.validate_resume(ck, cfg, world_seed, provenance)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), streams.load_state(ck["streams"])
        carry, cache, st = ck["carry"], ck["cache"], ck["state"]
        torch.set_rng_state(ck["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        controller.load_state(ck["fixation_controller"])
        print(f"resumed at update {st['update']}, frames {st['frames']}", flush=True)
    total, every = cfg["p1"]["frames"], cfg["p1"]["frames"] * cfg["p1"]["checkpoint_every_fraction"]
    log = open(out_dir / "log.jsonl", "a")
    t0, u0 = time.time(), st["update"]
    while st["frames"] < total and (updates is None or st["update"] < updates):
        S = steps or (d["chunk_steps"][0] if st["frames"] < total / 2 else d["chunk_steps"][1])
        micro = d["accumulate_at_64"] if S == d["chunk_steps"][0] else d["accumulate_at_256"]
        chunk = streams.next(S)
        controller.prepare(chunk, streams)
        at_checkpoint = checkpoints and st["frames"] + int(chunk["valid"][:, P:].sum()) >= st["next_ckpt"] * every
        f = TR.lr_factor(st["frames"], cfg)
        for g in opt.param_groups:
            g["lr"] = o["lr"] * g["lr_scale"] * f
        opt.zero_grad(set_to_none=True)
        rows_all = np.array_split(np.arange(d["streams"]), micro)
        boot = TR.bootstrap_rows(cfg, world_seed, st["update"])
        logs = []
        route_norms = None
        if at_checkpoint:
            diagnostic_carry = M.carry_rows(carry, [0])
            diagnostic_cache = cache[:1].clone()
        for rows in rows_all:
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                cache[rows], targets, cfg, boot[rows], st["update"])
            ((model.harmoniser(TR.groups(terms)) + terms["noise"]) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new
        if at_checkpoint:
            route_norms = TR.encoder_diagnostics(model, chunk, diagnostic_carry, diagnostic_cache, targets, cfg, boot, st["update"])
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]))
        opt.step()
        controller.learn(int(chunk["valid"][:, P:].sum()))
        st["frames"] += int(chunk["valid"][:, P:].sum())
        st["update"] += 1
        if st["update"] % log_every == 0 or st["update"] == 1:
            rec = {"update": st["update"], "frames": st["frames"], "steps": S, "lr": o["lr"] * f, "grad_norm": gn,
                   "sigma": model.harmoniser.log_sigma.exp().tolist(),
                   **{k: float(np.mean([l[k] for l in logs])) for k in logs[0]},
                   "sec_per_update": (time.time() - t0) / (st["update"] - u0),
                   "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            log.write(json.dumps(rec) + "\n"), log.flush()
            print(json.dumps(rec), flush=True)
            t0, u0 = time.time(), st["update"]
        if checkpoints and st["frames"] >= st["next_ckpt"] * every:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "streams": streams.state(),
                  "carry": carry, "cache": cache, "state": dict(st, next_ckpt=st["next_ckpt"] + 1),
                  "cfg": cfg, "world_seed": world_seed, "provenance": provenance,
                  "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}
            ck["fixation_controller"] = controller.state()
            torch.save(ck, out_dir / f"ckpt_{st['next_ckpt']:02d}.pt")
            torch.save(ck, latest)
            mon = TR.monitor(model, cfg, split["monitor"])
            log.write(json.dumps({"update": st["update"], "frames": st["frames"], "checkpoint": st["next_ckpt"],
                                  "monitor": mon, "encoder_grad_norms": route_norms}) + "\n"), log.flush()
            print("monitor", json.dumps(mon), flush=True)
            st["next_ckpt"] += 1
    controller.close()
    return model
