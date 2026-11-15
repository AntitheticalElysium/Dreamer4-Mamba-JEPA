"""F6.1 dissection, check 12 (PLAN.md): access or incentive? A substitution test: give the denoiser perfect, cheap,
per-cell recency (check 11's traces: per periphery cell, the squared change of its own coarse token, filtered at time
constants 8 and 32, through the previous frame) as a zero-initialised additive input to that cell's token, and
fine-tune the denoiser alone on G1 alone from F6.1 checkpoint 10 (F6.1's streams, carry and cache; AdamW 1e-4, 100
warm-up, clip 1; belief, perception and heads frozen). Read on the exact-history pairs with one generated frame (F5's
dissect.generate, the cache over the 8 identical frames, 32 draws): awake energy and food drops, the asleep onset pairs
(check 10c) and recovery, with each branch's own memory and traces, and with the traces exchanged.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.traces_ft      (GPU)
"""
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from saccade import chooser as C, data as D, model as M, train as TR

F61 = importlib.import_module("artifacts.experiments.20261114_p1f61.train")
DS = importlib.import_module("artifacts.experiments.20261111_p1f5.dissect")
TJ = importlib.import_module("artifacts.experiments.20261111_p1f5.trajectory")
SW = importlib.import_module("artifacts.experiments.20261115_f61_dissection.swap")
HERE = Path(__file__).resolve().parent
OUT = HERE / "run/traces"
CKPT = HERE.parent / "20261114_p1f61/run/ckpt_10.pt"
UPDATES, READS, N = 2000, (1000, 2000), 81
RATE = torch.tensor([1 / 8, 1 / 32])


def traces(p, first, e0, p0=None):
    """p [B, L, 81, 16] coarse tokens, first [B, L] episode starts, e0 [B, 81, 2] the state after the frame before frame
    0, p0 that frame's tokens (None: no earlier frame). Returns the input for each frame (the trace through the
    previous frame) [B, L, 81, 2] and the state after each frame [B, L, 81, 2]."""
    x = torch.zeros(p.shape[:3], device=p.device)
    x[:, 1:] = (p[:, 1:] - p[:, :-1]).float().square().sum(-1)
    if p0 is not None:
        x[:, 0] = (p[:, 0] - p0).float().square().sum(-1)
    x = x * (~first)[..., None]
    rate = RATE.to(p.device)
    e, inp, st = e0, [], []
    for i in range(p.shape[1]):
        e = torch.where(first[:, i, None, None], torch.zeros_like(e), e)
        inp.append(e)
        e = e + (x[:, i, :, None] - e) * rate
        st.append(e)
    return torch.stack(inp, 1), torch.stack(st, 1)


def install(den):
    """The traces enter each periphery cell's token through a zero-initialised linear map of log(1 + trace)."""
    den.trace_proj = nn.Linear(2, den.W).to(next(den.parameters()).device)
    nn.init.zeros_(den.trace_proj.weight); nn.init.zeros_(den.trace_proj.bias)
    embed = den.embed

    def with_traces(x, memory, fix):
        tok, ang = embed(x, memory, fix)
        tr = den.trace_input if den.trace_input.shape[0] == tok.shape[0] else den.trace_sub
        add = den.trace_proj(torch.log1p(tr.float())).to(tok.dtype)
        return torch.cat([tok[..., :N, :] + add, tok[..., N:, :]], -2), ang
    den.embed = with_traces
    loss = M.shortcut_loss

    def shortcut_loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels=None, noise=None):
        d.trace_sub = d.trace_input[bootstrap]                         # the bootstrap target's row subset
        return loss(d, x0, prev_action, memory, fix, episode, valid, sc, bootstrap, start, levels, noise)
    M.shortcut_loss = shortcut_loss


def pairs():
    """(name, rows as (prefix, frames, rewards, terms, q, actions [2, q + 1], truth [2, 2]), HUD slot, sign)."""
    out = []
    for name, slot, sign in (("energy", 66, -1), ("hunger", 64, -1), ("recovery", 63, +1)):
        rows = [(r[2], r[3], r[4], r[5], r[6], np.zeros((2, r[6] + 1), np.int64), r[7]) for r in TJ.pair_rows(name)]
        out.append((name, rows, slot, sign))
    onset = torch.load(SW.BANK, map_location="cpu", weights_only=False)
    rows = [((r["prefix_frames"], r["prefix_actions"], r["prefix_reward"], r["prefix_term"]), r["frames"], r["reward"], r["term"],
             r["end"], r["actions"][:, :r["end"] + 1], r["digit"][:, r["end"]:r["end"] + 2]) for r in onset]
    out.append(("asleep_onset", rows, 66, +1))
    return out


@torch.no_grad()
def pair_read(model, cfg, banks, readers):
    """dissect.generate with traces: cells (memory branch m, trace branch c); branch 0 is due."""
    den, dev, res = model.denoiser, DS.DEV, {}
    for name, rows, slot, sign in banks:
        rates = []
        for i, (prefix, frames, rewards, terms, q, acts, truth) in enumerate(rows):
            assert truth[0, 0] == truth[1, 0] and truth[0, 1] != truth[0, 0] and truth[1, 1] == truth[1, 0]
            old_f, old_a, old_r, old_d = prefix
            n = len(old_f)
            f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
            a = np.concatenate([np.repeat(old_a[None], 2, 0), acts], 1)
            rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
            td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
            fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
            query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=dev)
            act = torch.as_tensor(a, device=dev)
            prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
            ft = torch.as_tensor(f, device=dev)
            pc = torch.stack([C.coarse(model.perception, ft[b])[0] for b in (0, 1)])
            first = torch.zeros(pc.shape[:2], dtype=torch.bool, device=dev); first[:, 0] = True
            tin, tst = traces(pc, first, torch.zeros(2, N, 2, device=dev))
            with TR.autocast(dev, cfg):
                p = model.perception(ft, fix)
                y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                        torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
                bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
                assert all(torch.equal(z[0, -8:], z[1, -8:]) for z in p.parts()) and torch.equal(act[0, -9:], act[1, -9:])
                cell = {}
                for m, c in ((0, 0), (1, 1), (0, 1), (1, 0)):
                    rng = torch.Generator(device=dev).manual_seed(14000 + i)
                    cache = den.new_cache()
                    for t in range(n + q - 7, n + q + 1):
                        den.trace_input = tin[c:c + 1, t:t + 1]
                        cache = M.context_pass(den, tuple(z[0:1, t:t + 1] for z in p.parts()), prev[0:1, t:t + 1],
                                               model.belief.memory(bp[m:m + 1, t:t + 1]), fix[0:1, t:t + 1], cache, cfg, rng)
                    cache = {"t": cache["t"], "kv": [tuple(z.expand(DS.DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
                    den.trace_input = tst[c:c + 1, -1:].expand(DS.DRAWS, -1, -1, -1)
                    z = M.sample(den, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DS.DRAWS, 1),
                                 model.belief.memory(y[m:m + 1, -1:]).expand(DS.DRAWS, -1, -1, -1),
                                 query[None, None].expand(DS.DRAWS, 1, -1), cache, cfg, torch.Generator(device=dev).manual_seed(16000 + i))
                    digits = readers[slot][0](torch.cat([z[0][:, 0, slot], z[2][:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
                    cell[f"m{m}c{c}"] = float(((digits < truth[0, 0]) if sign < 0 else (digits > truth[0, 0])).mean())
            rates.append(cell)
        r = {k: np.array([x[k] for x in rates]) for k in rates[0]}
        ix = np.random.default_rng(1111).integers(len(rates), size=(10000, len(rates)))
        ci = lambda d: [float(d.mean()), *np.quantile(d[ix].mean(1), [.025, .975]).tolist()]
        res[name] = {"pairs": len(rates), "rate": {k: float(v.mean()) for k, v in r.items()},
                     "native_due_minus_delayed": ci(r["m0c0"] - r["m1c1"]),
                     "trace_effect": ci((r["m0c0"] + r["m1c0"] - r["m0c1"] - r["m1c1"]) / 2),
                     "memory_effect": ci((r["m0c0"] + r["m0c1"] - r["m1c0"] - r["m1c1"]) / 2)}
        print(name, json.dumps(res[name]), flush=True)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model, cfg = DS.NP.IF.load(CKPT.resolve())
    dev = next(model.parameters()).device
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    for p in model.parameters():
        p.requires_grad_(False)
    den = model.denoiser
    install(den)
    params = list(den.parameters())
    for p in params:
        p.requires_grad_(True)
    readers = TJ.readers(model, cfg, slots=(63, 64, 66))
    banks = pairs()
    res = {"checkpoint_sha256": TR.file_sha256(CKPT), "log": [], "reads": {}}
    model.eval()
    den.trace_input = torch.zeros(1, 1, N, 2, device=dev)
    res["reads"][0] = pair_read(model, cfg, banks, readers)             # update 0: the zero-initialised input is inert
    model.train()
    den.checkpoint = cfg["p1"]["activation_checkpointing"]
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=F61.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(M.MambaCarry(c.conv.to(dev), c.ssm.to(dev)) for c in ck["carry"])
    cache = ck["cache"].to(dev)
    base = ck["state"]["update"]
    e_prev, p_prev, tail = torch.zeros(d["streams"], N, 2, device=dev), None, None
    losses, t0 = [], time.time()
    for u in range(1, UPDATES + 1):
        chunk = streams.next(64)
        fr = torch.as_tensor(chunk["frames"], device=dev)
        with torch.no_grad():
            pc = torch.stack([C.coarse(model.perception, fr[b])[0] for b in range(len(fr))])
        first = torch.as_tensor(chunk["first"], device=dev)
        if tail is not None:                                            # the prefix repeats the previous chunk's last P frames
            assert torch.equal(fr[:, :P], tail) or u > 3
        tin, tst = traces(pc, first, e_prev, p_prev)
        T = fr.shape[1] - P
        e_prev, p_prev, tail = tst[:, T - 1], pc[:, T - 1], fr[:, T:]
        boot = TR.bootstrap_rows(cfg, F61.WORLD_SEED, base + u)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, u / 100)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            den.trace_input = tin[rows]
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                   None, cfg, boot[rows], base + u, None)
            (terms["g1"] / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            losses.append(float(terms["g1"]))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u == 1:
            print("peak GiB", torch.cuda.max_memory_allocated() / 2 ** 30, flush=True)
        if u % 100 == 0:
            res["log"].append({"update": u, "g1": float(np.mean(losses[-200:])), "trace_weight_norm": float(den.trace_proj.weight.norm()),
                               "sec_per_update": (time.time() - t0) / u})
            print(json.dumps(res["log"][-1]), flush=True)
        if u in READS:
            model.eval()
            res["reads"][u] = pair_read(model, cfg, banks, readers)
            torch.save({"denoiser": den.state_dict()}, OUT / f"denoiser_{u:04d}.pt")
            (HERE / "traces_ft.json").write_text(json.dumps(res, indent=1) + "\n")
            model.train()


if __name__ == "__main__":
    main()
