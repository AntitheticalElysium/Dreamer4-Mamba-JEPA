"""M0 training phases (saccade/design/M0_CODE.md; CURRICULUM.md; LOSSES.md). P1 = one-step world training on
episode streams with carried Mamba state (TBTT), diffusion forcing over chunks with an 8-frame context prefix, and
harmonised groups G1 (next percept), G2 (change map, TSP, fovea reconstruction), G3 (reward, continuation).
P2 / P3 are written after the S1 / S2 readings, against these interfaces.
Usage (project venv, GPU; long runs via systemd-run): python -m saccade.train p1 WORLD_SEED [--updates N] [--steps S]
  [--episodes N] [--out DIR] [--no-checkpoints]
"""
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from saccade import data as D, model as M

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "saccade/m0.yaml").read_text())


def set_reward_values(model: M.M0, episodes):
    """LOSSES G3: a sampled bin decodes to the mean of the train-split rewards falling in it (bin centre if none)."""
    r = torch.as_tensor(np.concatenate([D.RP.episode_fields(e.shard, e.slot)["rewards"].numpy() for e in episodes]))
    out = model.outcome
    k = out.bin_of(r.to(out.edges.device))
    total = torch.zeros(out.bins, dtype=torch.float64, device=k.device).index_add_(0, k, r.double().to(k.device))
    count = torch.bincount(k, minlength=out.bins)
    out.bin_value.copy_(torch.where(count > 0, total / count.clamp(min=1), out.bin_value.double()).float())


def lr_factor(frames: float, cfg) -> float:
    """Linear warm-up, then cosine to min_lr_ratio, both indexed by frames seen (CURRICULUM, Budgets)."""
    o, total = cfg["optimizer"], cfg["p1"]["frames"]
    if frames < o["warmup_frames"]:
        return frames / o["warmup_frames"]
    u = min(1.0, (frames - o["warmup_frames"]) / (total - o["warmup_frames"]))
    return o["min_lr_ratio"] + (1 - o["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * u))


def sample_sigma(shape, cfg, device, gen=None) -> torch.Tensor:
    s = cfg["edm"]["train_sigma"]
    z = torch.randn(shape, device=device, generator=gen)
    return (s["lognormal_loc"] + s["lognormal_scale"] * z).exp().clamp(s["min"], s["max"])


def autocast(dev, cfg):
    return torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda" and cfg["p1"]["precision"] == "bf16_autocast")


def tensors(chunk: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in chunk.items() if k != "target"}


def p1_losses(model: M.M0, chunk: dict, carry, cache, targets, cfg):
    """One microbatch of streams: chunk arrays [B, P + T, ...] (prefix = context only), carried Mamba state, cache =
    predicted beliefs [B, P + 1, W] of the prefix frames and of chunk frame 0 (computed in the previous chunk,
    detached; never replayed through Mamba). Returns (loss terms, new carry, new cache)."""
    P = cfg["data"]["context_prefix"]
    dev = cache.device
    t = tensors(chunk, dev)
    first, fix = t["first"], t["fixation"]
    L = first.shape[1]
    with autocast(dev, cfg):
        Pc = model.perception(t["frames"][:, P:], fix[:, P:])
        with torch.no_grad():
            Pp = model.perception(t["frames"][:, :P], fix[:, :P])
        emb = model.belief.percept_embed(Pc)
        x = model.belief.tokens(emb, t["reward"][:, P:], t["terminated"][:, P:], t["action"][:, P:])
        seq = first[:, P:].long().cumsum(1).repeat_interleave(2, 1)
        y, carry = model.belief(x, carry, seq)
        nxt = y[:, 1::2]                                                 # predicted belief of frame t + 1
        b_pred = torch.cat([cache.to(nxt.dtype), nxt[:, :-1]], 1) * (~first)[..., None]   # zero belief at resets
        new_cache = torch.cat([b_pred[:, -P:], nxt[:, -1:]], 1).detach().float()
        x0 = tuple(torch.cat([a, b.detach()], 1) for a, b in zip(Pp.parts(), Pc.parts()))
        sigma = sample_sigma(first.shape, cfg, dev)
        g1 = M.denoise_loss(model.denoiser, x0, sigma, t["prev_action"], model.belief.memory(b_pred), fix,
                            t["episode"], t["valid"], cfg["edm"]["sigma_data"])[:, P:].mean()
        bp, cells = b_pred[:, P:], M.window(fix[:, P:])
        pix = M.cell_pixels(M.crop(t["frames"][:, P:], fix[:, P:]))
        tsp = M.pixel_nll(*model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), Pc.v, cells), pix).mean()
        rec = M.pixel_nll(*model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), Pc.v, cells), pix).mean()
        logit = model.change(torch.cat([Pp.p, Pc.p], 1)[:, P - 1:L - 1], Pc.p, bp)
        idx = chunk["target"][:, P:]
        tgt = np.full((*idx.shape, 7, D.GRID), np.nan, np.float32)
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
    terms = {"g1": g1, "change": change, "tsp": tsp, "rec": rec, "reward": reward, "cont": cont}
    return {k: v.float() for k, v in terms.items()}, carry, new_cache


def groups(terms: dict) -> dict:
    """LOSSES terms table: G1 next percept; G2 grounding (change map, TSP, fovea reconstruction); G3 consequences."""
    return {"G1": terms["g1"], "G2": terms["change"] + terms["tsp"] + terms["rec"], "G3": terms["reward"] + terms["cont"]}


@torch.no_grad()
def monitor(model: M.M0, cfg, episodes, n_windows=None) -> dict:
    """The fixed P1 monitor (CURRICULUM, Budgets): the G1 loss of the target frame of fixed held-out windows (8 context
    frames at the inference context level, the target at 8 fixed noise levels = equal-mass quantiles of the training
    distribution), fixed noise draws, fixed fixations; beliefs from Mamba over each whole episode. Also token
    diagnostics (never gates): effective rank of the periphery tokens."""
    m, e = cfg["p1"]["monitor"], cfg["edm"]
    P, dev, was = cfg["data"]["context_prefix"], next(model.parameters()).device, model.training
    model.eval()
    lengths = np.array([ep.T for ep in episodes])
    rng = np.random.default_rng(m["fixed_noise_seed"])
    picks = rng.choice(int(lengths.sum()), n_windows or m["windows"], replace=False)
    off = np.r_[0, np.cumsum(lengths)]
    ep_of = np.searchsorted(off, picks, side="right") - 1
    z = torch.distributions.Normal(0, 1).icdf((torch.arange(m["sigma_levels"]) + 0.5) / m["sigma_levels"])
    s = e["train_sigma"]
    levels = (s["lognormal_loc"] + s["lognormal_scale"] * z).exp().clamp(s["min"], s["max"]).to(dev)
    losses, toks = [], []
    for i in np.unique(ep_of):
        ep = episodes[i]
        recs = [[(i, 1, t, *D.fixation_prior(np.random.default_rng([m["fixed_fixation_seed"], int(i), t]), ()))
                 for t in range(ep.T + 1)]]
        c = tensors(D.gather(episodes, recs), dev)
        with autocast(dev, cfg):
            Pe = M.Percept(*(torch.cat(z_, 1) for z_ in zip(*[
                model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                for a in range(0, ep.T + 1, 512)])), c["fixation"])
            toks.append(Pe.p[0, ::8].reshape(-1, Pe.p.shape[-1]).float())
            x = model.belief.tokens(model.belief.percept_embed(Pe), c["reward"], c["terminated"], c["action"])
            y, _ = model.belief(x, model.belief.initial(1, dev))
            b_pred = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            for t in (picks[ep_of == i] - off[i] + 1).tolist():                     # target frame t >= 1
                sl = slice(max(0, t - P), t + 1)
                n = sl.stop - sl.start
                rep = lambda a: a[:, sl].expand(len(levels), *a[:, sl].shape[1:])
                sig = torch.full((len(levels), n), e["inference_context_sigma"], device=dev)
                sig[:, -1] = levels
                g = torch.Generator(device=dev).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(i) * 4_099 + t)
                x0 = tuple(rep(a) for a in Pe.parts())
                noise = tuple(torch.randn(a.shape, device=dev, generator=g) for a in x0)
                l = M.denoise_loss(model.denoiser, x0, sig, rep(c["prev_action"]), model.belief.memory(rep(b_pred)),
                                   rep(c["fixation"]), rep(c["episode"]), rep(c["valid"]), e["sigma_data"], noise)
                losses.append(l[:, -1].float().cpu())
    L = torch.stack(losses)                                              # [windows, levels]
    tok = torch.cat(toks)
    ev = torch.linalg.eigvalsh(torch.cov(tok.T)).clamp(min=0)
    model.train(was)
    return {"g1": float(L.mean()), "g1_by_level": L.mean(0).tolist(), "levels": levels.tolist(),
            "periphery_effective_rank": float(ev.sum() ** 2 / (ev ** 2).sum()), "windows": len(L)}


def p1(cfg, world_seed, out_dir: Path, updates=None, steps=None, n_episodes=None, checkpoints=True, log_every=25):
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(world_seed)
    split = D.load_split(cfg)
    eps = split["train"][:n_episodes] if n_episodes else split["train"]
    targets = np.load(ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    offsets = D.frame_offsets(split["train"])
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    streams = D.Streams(eps, d["streams"], P, seed=world_seed, targets_offsets=offsets)
    model = M.M0(cfg).to(dev)
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    set_reward_values(model, split["train"])
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    W = cfg["belief"]["width"]
    st = {"frames": 0, "update": 0, "next_ckpt": 1}
    carry = model.belief.initial(d["streams"], dev)
    cache = torch.zeros(d["streams"], P + 1, W, device=dev)
    latest = out_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, weights_only=False, map_location=dev)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), streams.load_state(ck["streams"])
        carry, cache, st = ck["carry"], ck["cache"], ck["state"]
        torch.set_rng_state(ck["rng"]["cpu"].cpu()), torch.cuda.set_rng_state(ck["rng"]["cuda"].cpu())
        print(f"resumed at update {st['update']}, frames {st['frames']}", flush=True)
    total, every = cfg["p1"]["frames"], cfg["p1"]["frames"] * cfg["p1"]["checkpoint_every_fraction"]
    log = open(out_dir / "log.jsonl", "a")
    t0, u0 = time.time(), st["update"]
    while st["frames"] < total and (updates is None or st["update"] < updates):
        S = steps or (d["chunk_steps"][0] if st["frames"] < total / 2 else d["chunk_steps"][1])
        micro = 1 if S == d["chunk_steps"][0] else d["accumulate_at_256"]
        chunk = streams.next(S)
        f = lr_factor(st["frames"], cfg)
        for g in opt.param_groups:
            g["lr"] = o["lr"] * g["lr_scale"] * f
        opt.zero_grad(set_to_none=True)
        rows_all = np.array_split(np.arange(d["streams"]), micro)
        logs = []
        for rows in rows_all:
            terms, c_new, cache_new = p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                cache[rows], targets, cfg)
            (model.harmoniser(groups(terms)) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"]))
        opt.step()
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
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "streams": streams.state(), "carry": carry,
                  "cache": cache, "state": dict(st, next_ckpt=st["next_ckpt"] + 1), "cfg": cfg, "world_seed": world_seed,
                  "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}
            torch.save(ck, out_dir / f"ckpt_{st['next_ckpt']:02d}.pt")
            torch.save(ck, latest)
            mon = monitor(model, cfg, split["monitor"])
            log.write(json.dumps({"update": st["update"], "frames": st["frames"], "checkpoint": st["next_ckpt"],
                                  "monitor": mon}) + "\n"), log.flush()
            print("monitor", json.dumps(mon), flush=True)
            st["next_ckpt"] += 1
    return model


def main():
    a = argparse.ArgumentParser()
    a.add_argument("phase", choices=["p1"])
    a.add_argument("seed", type=int)
    a.add_argument("--updates", type=int)
    a.add_argument("--steps", type=int)
    a.add_argument("--episodes", type=int)
    a.add_argument("--out")
    a.add_argument("--no-checkpoints", action="store_true")
    args = a.parse_args()
    cfg = load_cfg()
    out = Path(args.out) if args.out else ROOT / cfg["p1"]["out_dir"] / f"p1_world{args.seed}"
    p1(cfg, args.seed, out, args.updates, args.steps, args.episodes, not args.no_checkpoints)


if __name__ == "__main__":
    main()
