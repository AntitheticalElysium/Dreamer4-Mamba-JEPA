"""M0 training phases (saccade/design/M0_CODE.md; CURRICULUM.md; LOSSES.md). P1 = one-step world training on
episode streams with carried Mamba state (TBTT), diffusion forcing over chunks with an 8-frame context prefix, and
harmonised groups G1 (next percept, shortcut forcing), G2 (change map, TSP, fovea reconstruction), G3 (reward,
continuation), with the learned fixation controller (saccade/chooser.py) choosing each target frame's glimpse. Canonical
since P1-F4 (11-07); earlier recipes run from commit fa6edd1a. P2 / P3 are written against these interfaces.
Usage (project venv, GPU; long runs via systemd-run): python -m saccade.train p1 WORLD_SEED [--updates N] [--steps S]
  [--episodes N] [--out DIR] [--no-checkpoints]
"""
import argparse, hashlib, json, math, subprocess, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from saccade import chooser as C, data as D, model as M

ROOT = Path(__file__).resolve().parents[1]


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "saccade/m0.yaml").read_text())


def file_sha256(path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_provenance(cfg) -> dict:
    """Small, content-based identity; unrelated repository commits may change without changing a run."""
    sources = ("saccade/model.py", "saccade/train.py", "saccade/data.py", "saccade/chooser.py", "saccade/m0.yaml")
    return {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {p: file_sha256(ROOT / p) for p in sources},
            "manifest_sha256": file_sha256(ROOT / cfg["data"]["corpus"] / "manifest.json"),
            "targets_sha256": file_sha256(ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy"),
            "torch": str(torch.__version__)}


def validate_resume(checkpoint, cfg, world_seed, provenance):
    """Fail before restoring any state if this is a different recipe, seed, source or corpus."""
    if checkpoint.get("cfg") != cfg or checkpoint.get("world_seed") != world_seed:
        raise ValueError("Resume configuration or world seed differs from the checkpoint")
    saved = checkpoint.get("provenance", {})
    for key in ("source_sha256", "manifest_sha256", "targets_sha256", "torch"):
        if saved.get(key) != provenance[key]:
            raise ValueError(f"Resume identity differs: {key}")


def encoder_grad_norms(model, terms):
    """Raw per-loss encoder gradient norms, without changing .grad or consuming training RNG."""
    params = tuple(model.perception.parameters())
    result = {}
    for name, loss in terms.items():
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        squares = [g.detach().float().square().sum() for g in grads if g is not None]
        result[name] = float(torch.stack(squares).sum().sqrt()) if squares else 0.0
    return result


def encoder_diagnostics(model, chunk, carry, cache, targets, cfg, bootstrap=None, update=None):
    """One stream, at most 64 new frames, after training backward. Preserve RNG and accumulated .grad.

    Reusing the full retained training graph for six extra backwards exceeded the 6 GB device budget.
    The bounded recomputation uses the same pre-update parameters and incoming history, not a new data draw.
    """
    n = cfg["data"]["context_prefix"] + 64
    bounded = {k: v[:1, :n] for k, v in chunk.items()}
    with torch.random.fork_rng(devices=[cache.device.index]):
        terms, _, _ = p1_losses(model, bounded, carry, cache, targets, cfg, bootstrap[:1], update)
        return encoder_grad_norms(model, terms)


def bootstrap_rows(cfg, world_seed, update) -> np.ndarray:
    """Shortcut forcing: the update's bootstrap (self-consistency) streams, a fraction of the batch as in both Dreamer 4
    reimplementations (self_fraction 0.25, round, at most B - 1). They pick the batch's last rows of random sequences;
    here streams persist, so a fresh random subset per update (its own seed: resumable, the training RNG untouched)."""
    n = cfg["data"]["streams"]
    k = max(0, min(n - 1, round(cfg["edm"]["shortcut"]["self_fraction"] * n)))
    rows = np.zeros(n, bool)
    rows[np.random.default_rng([world_seed, update]).choice(n, k, replace=False)] = True
    return rows


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


def autocast(dev, cfg):
    return torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda" and cfg["p1"]["precision"] == "bf16_autocast")


def tensors(chunk: dict, device) -> dict:
    return {k: torch.as_tensor(v, device=device) for k, v in chunk.items() if k != "target"}


def p1_losses(model: M.M0, chunk: dict, carry, cache, targets, cfg, bootstrap=None, update=None):
    """One microbatch of streams: chunk arrays [B, P + T, ...] (prefix = context only), carried Mamba state, cache =
    predicted beliefs [B, P + 1, W] of the prefix frames and of chunk frame 0 (computed in the previous chunk,
    detached; never replayed through Mamba), these streams' bootstrap rows (bootstrap_rows) and the update index (the
    bootstrap start). Returns (loss terms, new carry, new cache)."""
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
        sc = cfg["edm"]["shortcut"]                                      # Dreamer 4's shortcut forcing (P1-F4)
        g1 = M.shortcut_loss(model.denoiser, x0, t["prev_action"], model.belief.memory(b_pred), fix, t["episode"],
                             t["valid"], sc, torch.as_tensor(bootstrap, device=dev), update >= sc["bootstrap_start"])[:, P:].mean()
        bp, cells = b_pred[:, P:], M.window(fix[:, P:])
        pix = M.cell_pixels(M.crop(t["frames"][:, P:], fix[:, P:]))
        assert cfg["heads"]["pixel_decoder"]["mean_loss"] == "faithful"   # LOSSES G2 (10-31): Stirn et al. 2023
        vs = Pc.v.detach()                                               # the scale's lighting input (the mean never reads it)
        mt, st = model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), vs, cells)
        mr, sr = model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), vs, cells)
        y = pix.float() / 127.5 - 1
        tsp, rec = (mt.float() - y).square().mean(), (mr.float() - y).square().mean()
        noise = M.pixel_nll(mt.detach(), st, pix).mean() + M.pixel_nll(mr.detach(), sr, pix).mean()
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
    terms = {"g1": g1, "change": change, "tsp": tsp, "rec": rec, "reward": reward, "cont": cont,
             "noise": noise}                                             # noise reaches only the scale network: outside the groups
    return {k: v.float() for k, v in terms.items()}, carry, new_cache


def groups(terms: dict) -> dict:
    """LOSSES terms table: G1 next percept; G2 grounding (change map, TSP, fovea reconstruction); G3 consequences."""
    return {"G1": terms["g1"], "G2": terms["change"] + terms["tsp"] + terms["rec"], "G3": terms["reward"] + terms["cont"]}


@torch.no_grad()
def monitor(model: M.M0, cfg, episodes, n_windows=None) -> dict:
    """Fixed held-out reading, with temporary raw-token diagnostics and no training-state changes."""
    stats = {}

    def capture(name):
        def hook(module, inputs, output):
            x = output.float()
            if name == "fovea":
                x = x.movedim(1, -1)
            x = x.reshape(-1, x.shape[-1])
            total, squares = x.sum(0).double(), x.square().sum(0).double()
            if name not in stats:
                stats[name] = [0, torch.zeros_like(total), torch.zeros_like(squares)]
            stats[name][0] += len(x)
            stats[name][1] += total
            stats[name][2] += squares
        return hook

    hooks = [module.register_forward_hook(capture(name)) for name, module in
             (("periphery", model.perception.p_out), ("fovea", model.perception.fovea), ("lighting", model.perception.light))]
    was = model.training
    model.eval()
    try:
        result = _monitor(model, cfg, episodes, n_windows)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was)
    result["raw_token_statistics"] = {
        name: {"tokens": n, "mean_channel_std": float((sq / n - (s / n).square()).clamp(min=0).sqrt().mean()),
               "rms": float((sq.mean() / n).sqrt())} for name, (n, s, sq) in stats.items()}
    return result


@torch.no_grad()
def _monitor(model: M.M0, cfg, episodes, n_windows=None) -> dict:
    """The fixed P1 monitor (CURRICULUM, Budgets): the flow term of G1 (ramp-weighted x-space error) on the target frame
    of fixed held-out windows at every finest-grid signal level (8 context frames at the context signal level), fixed
    noise draws, fixed fixations; beliefs from Mamba over each whole episode. Also token diagnostics (never gates):
    effective rank of the periphery tokens."""
    m, e = cfg["p1"]["monitor"], cfg["edm"]
    P, dev = cfg["data"]["context_prefix"], next(model.parameters()).device
    lengths = np.array([ep.T for ep in episodes])
    rng = np.random.default_rng(m["fixed_noise_seed"])
    picks = rng.choice(int(lengths.sum()), n_windows or m["windows"], replace=False)
    off = np.r_[0, np.cumsum(lengths)]
    ep_of = np.searchsorted(off, picks, side="right") - 1
    k = e["shortcut"]["k_max"]
    levels = torch.arange(k, device=dev)
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
                g = torch.Generator(device=dev).manual_seed(m["fixed_noise_seed"] * 1_000_003 + int(i) * 4_099 + t)
                x0 = tuple(rep(a) for a in Pe.parts())
                noise = tuple(torch.randn(a.shape, device=dev, generator=g) for a in x0)
                sg = torch.full((len(levels), n), e["shortcut"]["context_signal"], device=dev, dtype=torch.long)
                sg[:, -1] = levels
                lv = [(sg, torch.full_like(sg, k.bit_length() - 1))] * 2
                l = M.shortcut_loss(model.denoiser, x0, rep(c["prev_action"]), model.belief.memory(rep(b_pred)),
                                    rep(c["fixation"]), rep(c["episode"]), rep(c["valid"]), e["shortcut"],
                                    torch.zeros(len(levels), dtype=torch.bool, device=dev), False, lv, noise)
                losses.append(l[:, -1].float().cpu())
    L = torch.stack(losses)                                              # [windows, levels]
    tok = torch.cat(toks)
    ev = torch.linalg.eigvalsh(torch.cov(tok.T)).clamp(min=0)
    return {"g1": float(L.mean()), "g1_by_level": L.mean(0).tolist(),
            "levels": (levels.float() / k).tolist(),
            "periphery_effective_rank": float(ev.sum() ** 2 / (ev ** 2).sum()), "windows": len(L)}


def p1(cfg, world_seed, out_dir: Path, updates=None, steps=None, n_episodes=None, checkpoints=True, log_every=25):
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(world_seed)
    split = D.load_split(cfg)
    eps = split["train"][:n_episodes] if n_episodes else split["train"]
    targets = np.load(ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    offsets = D.frame_offsets(split["train"])
    if targets.shape != (int(offsets[-1]), 7, D.GRID):
        raise ValueError("Change targets do not match the configured training split")
    provenance = run_provenance(cfg)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    assert max(d["accumulate_at_64"], d["accumulate_at_256"]) <= d["streams"], "a microbatch without streams"
    streams = D.Streams(eps, d["streams"], P, seed=world_seed, targets_offsets=offsets)
    model = M.M0(cfg).to(dev)
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    set_reward_values(model, split["train"])
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    controller = C.Controller(model, cfg, out_dir, dev)
    W = cfg["belief"]["width"]
    st = {"frames": 0, "update": 0, "next_ckpt": 1}
    carry = model.belief.initial(d["streams"], dev)
    cache = torch.zeros(d["streams"], P + 1, W, device=dev)
    latest = out_dir / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, weights_only=False, map_location=dev)
        validate_resume(ck, cfg, world_seed, provenance)
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
        f = lr_factor(st["frames"], cfg)
        for g in opt.param_groups:
            g["lr"] = o["lr"] * g["lr_scale"] * f
        opt.zero_grad(set_to_none=True)
        rows_all = np.array_split(np.arange(d["streams"]), micro)
        boot = bootstrap_rows(cfg, world_seed, st["update"])
        logs = []
        route_norms = None
        if at_checkpoint:
            diagnostic_carry = M.carry_rows(carry, [0])
            diagnostic_cache = cache[:1].clone()
        for rows in rows_all:
            terms, c_new, cache_new = p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows),
                                                cache[rows], targets, cfg, boot[rows], st["update"])
            ((model.harmoniser(groups(terms)) + terms["noise"]) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            logs.append({k: float(v.detach()) for k, v in terms.items()})
        del terms, c_new, cache_new
        if at_checkpoint:
            route_norms = encoder_diagnostics(model, chunk, diagnostic_carry, diagnostic_cache, targets, cfg, boot, st["update"])
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
            ck = {"model": model.state_dict(), "opt": opt.state_dict(), "streams": streams.state(), "carry": carry,
                  "cache": cache, "state": dict(st, next_ckpt=st["next_ckpt"] + 1), "cfg": cfg, "world_seed": world_seed,
                  "provenance": provenance,
                  "rng": {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}}
            ck["fixation_controller"] = controller.state()
            torch.save(ck, out_dir / f"ckpt_{st['next_ckpt']:02d}.pt")
            torch.save(ck, latest)
            mon = monitor(model, cfg, split["monitor"])
            log.write(json.dumps({"update": st["update"], "frames": st["frames"], "checkpoint": st["next_ckpt"],
                                  "monitor": mon, "encoder_grad_norms": route_norms}) + "\n"), log.flush()
            print("monitor", json.dumps(mon), flush=True)
            st["next_ckpt"] += 1
    controller.close()
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
