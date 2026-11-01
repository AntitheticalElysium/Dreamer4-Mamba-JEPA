"""Fovea probes F1 (retention) and F2 (value of looking) on the frozen P1 world seed 1, checkpoint 10; CPU only.

Claims, competing explanations, controls and readouts: PLAN.md (this folder), written before this script ran.
F1 per frame of each processed episode: TSP NLL and fovea-reconstruction NLL per glimpse cell (nats per pixel), with
the recorded random fixations (S1's seeded prior). F2 at sampled frames t: the 49 counterfactual glimpses of the real
frame t enter the belief (predicted belief of t + 1) and the denoiser context; the G1 loss of the real frame t + 1 on
its periphery (map rows / HUD rows) and lighting tokens at the 8 monitor noise levels, common random numbers across the
49 fixations, two independent noise sets. Also, per counterfactual fixation, the TSP - reconstruction NLL gap at t.
Usage (project venv, CPU): python -m artifacts.experiments.20261029_fovea_probes.measure OUT_DIR N_EPISODES
"""
import math, os, sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
DEV = torch.device("cpu")
P = 8
CENTRES = torch.tensor([(r, c) for r in range(1, 8) for c in range(1, 8)])
MAP_CELLS = 63                                                           # screen rows 0-6 = map, 7-8 = HUD
RANDOM_PER_EP, NIGHT_PER_EP = 4, 2


def levels(cfg):
    m, s = cfg["p1"]["monitor"], cfg["edm"]["train_sigma"]
    z = torch.distributions.Normal(0, 1).icdf((torch.arange(m["sigma_levels"]) + 0.5) / m["sigma_levels"])
    return (s["lognormal_loc"] + s["lognormal_scale"] * z).exp().clamp(s["min"], s["max"]).tolist()


def expand(cache, n):
    return {"t": cache["t"], "kv": [(k.expand(n, *k.shape[1:]).contiguous(), v.expand(n, *v.shape[1:]).contiguous())
                                    for k, v in cache["kv"]]}


def enter(den, x, noise, prev_action, memory, fix, cache, cfg):
    """model.context_pass with a given context-noise draw (common across rows)."""
    sc, sd = cfg["edm"]["inference_context_sigma"], cfg["edm"]["sigma_data"]
    xs = tuple(t + sc * n for t, n in zip(x, noise))
    c_skip, c_out, c_in, c_noise = M.precondition(torch.full((x[0].shape[0], 1), sc), sd)
    _, new = den.step(tuple(M._bcast(c_in, t) * t for t in xs), c_noise, prev_action, memory, fix, cache)
    return den.commit(cache, new)


def pixel_heads(model, bp, Pc, frames, fix):
    """Per glimpse cell: TSP NLL (from predicted belief, periphery, lighting) and reconstruction NLL (from the glimpse)."""
    cells = M.window(fix)
    pix = M.cell_pixels(M.crop(frames, fix))
    tsp = M.pixel_nll(*model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), Pc.v, cells), pix).mean(-1)
    rec = M.pixel_nll(*model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), Pc.v, cells), pix).mean(-1)
    return tsp, rec


def samples(L, e, s_off, T, rng):
    """F2 frames of episode e: t in [P, T - 1] (t + 1 exists). Random, and night with any mob in view (separate)."""
    cand = np.arange(P, T)
    if len(cand) == 0:
        return [], []
    g = s_off[e] + cand
    pos = L["s_pos"][g].astype(int)
    mob = np.zeros(len(cand), bool)
    for k in ("zombies", "cows", "skeletons", "arrows"):
        rel = L[f"s_{k}_p"][g].astype(int) - pos[:, None]
        mob |= (L[f"s_{k}_m"][g] & (np.abs(rel[..., 0]) <= 3) & (np.abs(rel[..., 1]) <= 4)).any(1)
    rand = rng.choice(cand, min(RANDOM_PER_EP, len(cand)), replace=False)
    night = cand[(L["s_light"][g] < 0.5) & mob & ~np.isin(cand, rand)]
    return list(rand), list(rng.choice(night, min(NIGHT_PER_EP, len(night)), replace=False)) if len(night) else []


@torch.no_grad()
def episode(model, cfg, L, e, ep, s_off, sig, rng, gen):
    fr_rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
    recs = [[(0, 1, t, *D.fixation_prior(fr_rng, ())) for t in range(ep.T + 1)]]
    c = TR.tensors(D.gather([ep], recs), DEV)
    parts = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
    Pc = M.Percept(*(torch.cat(z, 1) for z in zip(*[q.parts() for q in parts])), c["fixation"])
    emb = model.belief.percept_embed(Pc)
    tok = model.belief.tokens(emb, c["reward"], c["terminated"], c["action"])
    rand, night = samples(L, e, s_off, ep.T, rng)
    cuts = sorted(set(rand) | set(night))
    ys, carries, carry, at = [], {}, model.belief.initial(1, DEV), 0
    for t in cuts + [ep.T + 1]:
        y, carry = model.belief(tok[:, at:2 * t], carry, backend="reference")
        ys.append(y); at = 2 * t
        carries[t] = carry
    y = torch.cat(ys, 1)
    bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)  # predicted belief of each frame
    tsp, rec = pixel_heads(model, bp, Pc, c["frames"], c["fixation"])
    f1 = {"tsp": tsp[0].numpy().astype(np.float32), "rec": rec[0].numpy().astype(np.float32),
          "fix": c["fixation"][0].numpy().astype(np.int8)}
    den, n = model.denoiser, len(CENTRES)
    out = []
    for t in cuts:
        cache = den.new_cache()
        for dt in range(-P, 0):
            s = t + dt
            x = tuple(a[:, s:s + 1] for a in (Pc.p, Pc.g, Pc.v))
            noise = tuple(torch.randn(a.shape, generator=gen) for a in x)
            cache = enter(den, x, noise, c["prev_action"][:, s:s + 1], model.belief.memory(bp[:, s:s + 1]),
                          c["fixation"][:, s:s + 1], cache, cfg)
        cache = expand(cache, n)
        fr = c["frames"][:, t].expand(n, -1, -1, -1)
        g_L = model.perception(fr, CENTRES).g                              # [49, 9, 64]; p, v do not depend on fix
        Pt = M.Percept(Pc.p[:, t].expand(n, -1, -1), g_L, Pc.v[:, t].expand(n, -1), CENTRES)
        tL, rL = pixel_heads(model, bp[:, t].expand(n, -1), Pt, fr, CENTRES)
        e_t = model.belief.percept_embed(Pt)[:, None]
        tk = model.belief.tokens(e_t, c["reward"][:, t].expand(n)[:, None], c["terminated"][:, t].expand(n)[:, None],
                                 c["action"][:, t].expand(n)[:, None])
        carry_n = tuple(M.MambaCarry(q.conv.expand(n, *q.conv.shape[1:]).contiguous(),
                                     q.ssm.expand(n, *q.ssm.shape[1:]).contiguous()) for q in carries[t])
        yL, _ = model.belief(tk, carry_n, backend="reference")
        bp_next = yL[:, 1]                                                  # predicted belief of t + 1 under each L
        x = (Pt.p[:, None], g_L[:, None], Pt.v[:, None])
        noise = tuple(torch.randn((1, *a.shape[1:]), generator=gen).expand_as(a) for a in x)
        cache = enter(den, x, noise, c["prev_action"][:, t:t + 1].expand(n, 1), model.belief.memory(bp[:, t:t + 1]).expand(n, -1, -1, -1),
                      CENTRES[:, None], cache, cfg)
        x0 = tuple(a[:, t + 1].expand(n, *a.shape[2:])[:, None] for a in (Pc.p, Pc.g, Pc.v))
        pa, fx = c["prev_action"][:, t + 1:t + 2].expand(n, 1), c["fixation"][:, t + 1:t + 2].expand(n, 1, 2)
        mem = model.belief.memory(bp_next)[:, None]
        loss = np.zeros((2, len(sig), n, 3), np.float32)
        for a_set in range(2):
            for k, s in enumerate(sig):
                eps = tuple(torch.randn((1, *a.shape[1:]), generator=gen).expand_as(a) for a in x0)
                xs = tuple(a + s * z for a, z in zip(x0, eps))
                c_skip, c_out, c_in, c_noise = M.precondition(torch.full((n, 1), s), cfg["edm"]["sigma_data"])
                Fo, _ = den.step(tuple(M._bcast(c_in, a) * a for a in xs), c_noise, pa, mem, fx, cache)
                se = [((f.float() - (a - M._bcast(c_skip, a) * z) / M._bcast(c_out, a)) ** 2) for f, a, z in zip(Fo, x0, xs)]
                loss[a_set, k, :, 0] = se[0][:, 0, :MAP_CELLS].sum((-1, -2)).numpy()
                loss[a_set, k, :, 1] = se[0][:, 0, MAP_CELLS:].sum((-1, -2)).numpy()
                loss[a_set, k, :, 2] = se[2][:, 0].sum(-1).numpy()
        out.append({"e": e, "t": t, "night_set": t in night, "loss": loss,
                    "tsp_L": tL.mean(-1).numpy(), "rec_L": rL.mean(-1).numpy()})
    return f1, out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    n_ep = int(sys.argv[2])
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg); model.load_state_dict(ck["model"]); model.eval()
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    sig = levels(cfg)
    rng, gen = np.random.default_rng(2029), torch.Generator().manual_seed(2029)
    F1, F2 = [], []
    for e, ep in enumerate(S.episodes(L)):
        f1, f2 = episode(model, cfg, L, e, ep, s_off, sig, rng, gen)
        F1.append(f1); F2 += f2
        if e % 10 == 9 or e == len(L["ep_T"]) - 1:
            np.savez_compressed(out_dir / "f1.npz", **{k: np.concatenate([f[k] for f in F1]) for k in F1[0]},
                                ep_T=L["ep_T"][:e + 1])
            np.savez_compressed(out_dir / "f2.npz", **{k: np.stack([s[k] for s in F2]) for k in F2[0]}, sigmas=np.array(sig))
            print(f"episode {e + 1}/{len(L['ep_T'])}, F2 samples {len(F2)}", flush=True)


if __name__ == "__main__":
    main()
