"""P1-MultiStep-Mamba trainer: P1-HUD-Change with Multi-Step Mamba belief self-consistency."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from saccade import chooser as C, data as D, model as M, train as TR
from . import model_multistep as MM

HORIZONS = (2, 4, 8, 16)
LAMBDA_MULTI = 0.1


def multistep_losses(model: MM.MultiStepM0, chunk: dict, carry, cache, targets, cfg, bootstrap=None, update=0, gen=None):
    """Teacher pass over the chunk + blind student multi-step rollout from sampled prefix t over horizon H."""
    dev = next(model.parameters()).device
    P = cfg["data"]["context_prefix"]
    t = TR.tensors(chunk, dev)
    first, fix = t["first"], t["fixation"]
    L = first.shape[1]
    T_chunk = L - P

    # Sample horizon H and starting frame t
    # Horizons from {2, 4, 8, 16}
    h_idx = int(torch.randint(0, len(HORIZONS), (), generator=gen))
    H = HORIZONS[h_idx]
    assert T_chunk > H + 1, f"Chunk length {T_chunk} too short for horizon {H}"
    start_t = int(torch.randint(0, T_chunk - H, (), generator=gen))

    with TR.autocast(dev, cfg):
        Pc = model.perception(t["frames"][:, P:], fix[:, P:])
        with torch.no_grad():
            Pp = model.perception(t["frames"][:, :P], fix[:, :P])
        emb = model.belief.percept_embed(Pc)
        x = model.belief.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:])
        seq = first[:, P:].long().cumsum(1).repeat_interleave(2, 1)

        # Split forward pass at start_t immediately after percept t (split_idx = 2 * start_t + 1)
        split_idx = 2 * start_t + 1
        y_hist, carry_t = model.belief(x[:, :split_idx], carry, seq[:, :split_idx])
        y_rest, c_new = model.belief(x[:, split_idx:], carry_t, seq[:, split_idx:])
        y = torch.cat([y_hist, y_rest], dim=1)

        # Student rollout from carry_t (un-detached: gradients flow into history!)
        # Process action t first, then interleave mask_token and actions up to t + H
        act_span = t["action"][:, P + start_t : P + start_t + H]          # [B, H]
        student_tokens = model.belief.rollout_tokens(act_span)            # [B, 2*H, W]
        y_student, _ = model.belief(student_tokens, carry_t)
        b_student = y_student[:, -1]                                      # student's post-mask belief at t+H

        # Target: teacher's observation-informed belief at t+H (token index 2 * (start_t + H))
        b_target = y[:, 2 * (start_t + H)].detach()

        # Episode boundary safety: do not cross resets
        valid_rollout = (~first[:, P + start_t : P + start_t + H + 1].any(dim=1))
        if valid_rollout.any():
            multi_err = ((b_student - b_target) ** 2).mean(-1)
            multi_loss = multi_err[valid_rollout].mean()
        else:
            multi_loss = torch.tensor(0.0, device=dev)

        # Standard F4 / HUD-change losses (G1, G2, G3)
        nxt = y[:, 1::2]                                                 # predicted belief of frame t + 1
        b_pred = torch.cat([cache.to(nxt.dtype), nxt[:, :-1]], 1) * (~first)[..., None]
        new_cache = torch.cat([b_pred[:, -P:], nxt[:, -1:]], 1).detach().float()
        x0 = tuple(torch.cat([a, b.detach()], 1) for a, b in zip(Pp.parts(), Pc.parts()))
        sc = cfg["edm"]["shortcut"]
        g1 = M.shortcut_loss(model.denoiser, x0, t["prev_action"], model.belief.memory(b_pred), fix, t["episode"],
                             t["valid"], sc, torch.as_tensor(bootstrap, device=dev), update >= sc["bootstrap_start"])[:, P:].mean()
        bp, cells = b_pred[:, P:], M.window(fix[:, P:])
        pix = M.cell_pixels(M.crop(t["frames"][:, P:], fix[:, P:]))
        assert cfg["heads"]["pixel_decoder"]["mean_loss"] == "faithful"
        vs = Pc.v.detach()
        mt, st = model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), vs, cells)
        mr, sr = model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), vs, cells)
        y_pix = pix.float() / 127.5 - 1
        tsp, rec = (mt.float() - y_pix).square().mean(), (mr.float() - y_pix).square().mean()
        noise = M.pixel_nll(mt.detach(), st, pix).mean() + M.pixel_nll(mr.detach(), sr, pix).mean()
        logit = model.change(torch.cat([Pp.p, Pc.p], 1)[:, P - 1:L - 1], Pc.p, bp)
        idx = chunk["target"][:, P:]
        tgt = np.full((*idx.shape, D.GRID, D.GRID, len(D.CHANGE_MODES)), np.nan, np.float32)
        tgt[idx >= 0] = targets[idx[idx >= 0]] if targets is not None else np.nan
        tgt = torch.as_tensor(tgt, device=dev)
        ok = torch.isfinite(tgt)
        change = F.binary_cross_entropy_with_logits(logit.float()[ok], tgt[ok]) if ok.any() else logit.sum() * 0
        nf = ~first[:, P:]
        h = model.outcome.features(bp, emb)[nf]
        r = t["reward"][:, P:][nf]
        reward = -model.outcome.reward_logp(h, r).mean()
        cont = F.binary_cross_entropy_with_logits(model.outcome.cont_logit(h, r).float(),
                                                  (~t["terminated"][:, P:][nf]).float())

    terms = {"g1": g1, "change": change, "tsp": tsp, "rec": rec, "reward": reward, "cont": cont,
             "noise": noise, "multi": multi_loss}
    return {k: v.float() for k, v in terms.items()}, c_new, new_cache


def p1(cfg, world_seed, out_dir: Path, updates=None, steps=None, n_episodes=None, checkpoints=True, log_every=25, controller_cfg=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(world_seed)
    gen = torch.Generator(device="cpu").manual_seed(world_seed + 1000)
    split = D.load_split(cfg)
    eps = split["train"][:n_episodes] if n_episodes else split["train"]
    targets = np.load(D.targets_dir(cfg) / "change_targets.npy", mmap_mode="r")
    offsets = D.frame_offsets(split["train"])
    D.validate_change_targets(targets, split["train"], D.targets_dir(cfg))
    provenance = TR.run_provenance(cfg)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    assert max(d["accumulate_at_64"], d["accumulate_at_256"]) <= d["streams"], "a microbatch without streams"
    streams = D.Streams(eps, d["streams"], P, seed=world_seed, targets_offsets=offsets)
    model = MM.MultiStepM0(cfg).to(dev)
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
        if "gen" in ck["rng"]:
            gen.set_state(ck["rng"]["gen"].cpu())
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
            terms, c_new, cache_new = multistep_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                       cache[rows], targets, cfg, boot[rows], st["update"], gen)
            # Total loss: existing harmonised groups + noise + modest fixed weight on multi-step belief loss
            f4_loss = model.harmoniser(TR.groups(terms)) + terms["noise"]
            total_loss = (f4_loss + LAMBDA_MULTI * terms["multi"]) / micro
            total_loss.backward()
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
                   "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "gen": gen.get_state()}}
            ck["fixation_controller"] = controller.state()
            torch.save(ck, out_dir / f"ckpt_{st['next_ckpt']:02d}.pt")
            torch.save(ck, latest)
            mon = TR.monitor(model, cfg, split["monitor"])
            log.write(json.dumps({"update": st["update"], "frames": st["frames"], "checkpoint": st["next_ckpt"],
                                  "monitor": mon, "encoder_grad_norms": route_norms}) + "\n"), log.flush()
            print("monitor", json.dumps(mon), flush=True)
            st["next_ckpt"] += 1
    return model
