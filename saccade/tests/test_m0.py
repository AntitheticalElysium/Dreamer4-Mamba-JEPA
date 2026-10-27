"""M0 checks before any long run (saccade/design/M0_CODE.md, tests section). GPU tests are skipped without CUDA.
Run: .venv/bin/python -m pytest saccade/tests -q
"""
import copy, math

import numpy as np
import pytest
import torch

from saccade import data as D, model as M, train as TR

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA (Triton Mamba kernel)")


@pytest.fixture(autouse=True)
def free_gpu():
    yield
    import gc
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def cfg():
    return TR.load_cfg()


@pytest.fixture(scope="module")
def split(cfg):
    return D.load_split(cfg)


# ---- data ----

def test_streams_follow_episodes_and_are_deterministic(cfg, split):
    eps = split["train"][:6]
    a, b = D.Streams(eps, 3, 8, seed=5), D.Streams(eps, 3, 8, seed=5)
    prev = None
    for steps in (40, 64, 64, 64):
        ca, cb = a.next(steps), b.next(steps)
        for k in ca:
            assert np.array_equal(ca[k], cb[k]), k
        if prev is not None:                                            # the prefix is the previous chunk's tail
            for k in ("frames", "fixation", "episode", "action", "reward", "first"):
                assert np.array_equal(ca[k][:, :8], prev[k][:, -8:]), k
        prev = ca
        for i in range(3):
            v = ca["valid"][i]
            ep, first = ca["episode"][i][v], ca["first"][i][v]
            assert np.all((np.diff(ep) == 0) | first[1:])               # a new episode only at a first frame
    # alignment against the store: o_t arrives with r_{t-1}, terminated_{t-1}; a_t is the action taken at t
    recs = [[(0, 1, t, 4, 4) for t in range(eps[0].T + 1)]]
    c = D.gather(eps, recs)
    f = D.RP.episode_fields(eps[0].shard, eps[0].slot)
    assert np.array_equal(c["reward"][0, 1:], f["rewards"].numpy()) and c["reward"][0, 0] == 0
    assert np.array_equal(c["terminated"][0, 1:], f["terminated"].numpy())
    assert np.array_equal(c["action"][0, :-1], f["actions_taken"].numpy()) and c["action"][0, -1] == 0
    assert np.array_equal(c["prev_action"][0, 1:], f["actions_taken"].numpy()) and c["prev_action"][0, 0] == -1
    assert np.array_equal(c["frames"][0], f["observations"].numpy())


# ---- perception ----

def test_perception_is_causal_frame_local_and_bounded(cfg):
    torch.manual_seed(0)
    per = M.Perception(cfg).eval()
    frames = torch.randint(0, 256, (2, 5, 63, 63, 3), dtype=torch.uint8)
    fix = torch.randint(1, 8, (2, 5, 2))
    with torch.no_grad():
        P = per(frames, fix)
        alt = frames.clone()
        alt[:, 3:] = torch.randint(0, 256, alt[:, 3:].shape, dtype=torch.uint8)   # future frames
        alt[1] = 0                                                                  # another episode / padding
        Q = per(alt, fix)
        one = per(frames[0, 2], fix[0, 2])
    for a, b in zip(P.parts(), Q.parts()):
        assert torch.equal(a[0, :3], b[0, :3])
    for a, b in zip(P.parts(), one.parts()):
        assert torch.allclose(a[0, 2], b, atol=1e-5)
    assert max(float(t.abs().max()) for t in P.parts()) < 3.0


def test_periphery_is_equivariant_to_whole_cell_shifts(cfg):
    torch.manual_seed(0)
    per = M.Perception(cfg).eval()
    frame = torch.randint(0, 256, (63, 63, 3), dtype=torch.uint8)
    shifted = torch.roll(frame, M.CELL, dims=1)                         # a permutation: pooled statistics unchanged
    fix = torch.tensor([4, 4])
    with torch.no_grad():
        p, q = per(frame, fix).p.reshape(9, 9, -1), per(shifted, fix).p.reshape(9, 9, -1)
    assert torch.allclose(torch.roll(p, 1, dims=1), q, atol=1e-5)


# ---- Mamba belief ----

def _tokens(B, L, W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, L, W, generator=g)


@gpu
def test_mamba_chunks_steps_and_resets(cfg):
    torch.manual_seed(0)
    bel = M.Belief(cfg).cuda()
    W = cfg["belief"]["width"]
    x = _tokens(2, 24, W).cuda()
    c0 = tuple(M.MambaCarry(torch.randn_like(c.conv), torch.randn_like(c.ssm)) for c in bel.initial(2, "cuda"))
    with torch.no_grad():
        full, cf = bel(x, c0)
        a, ca = bel(x[:, :10], c0)
        b, cb = bel(x[:, 10:], ca)
        assert torch.allclose(torch.cat([a, b], 1), full, atol=2e-3)                 # chunked = contiguous
        steps, c = [], c0
        for t in range(24):
            y, c = bel(x[:, t:t + 1], c)
            steps.append(y)
        assert torch.allclose(torch.cat(steps, 1), full, atol=2e-3)                  # step-by-step = scan
        # an episode start at token 9 (local seq_idx 0 before, 1 after) = carried scan before, fresh scan after
        seq = (torch.arange(24, device="cuda") >= 9).long().expand(2, -1).contiguous()
        y, cr = bel(x, c0, seq)
        before, _ = bel(x[:, :9], c0)
        after, cafter = bel(x[:, 9:], bel.initial(2, "cuda"))
        assert torch.allclose(y[:, :9], before, atol=2e-3) and torch.allclose(y[:, 9:], after, atol=2e-3)
        for u, v in zip(cr, cafter):                                                  # carry = the new episode's
            assert torch.allclose(u.conv, v.conv, atol=1e-5) and torch.allclose(u.ssm, v.ssm, atol=2e-3)
        # a chunk starting with an episode start (first index 1) drops the carried state
        y1, _ = bel(x, c0, torch.ones(2, 24, dtype=torch.long, device="cuda"))
        y0, _ = bel(x, bel.initial(2, "cuda"))
        assert torch.allclose(y1, y0, atol=2e-3)
        # the pure-torch reference agrees with the kernel, resets included
        yr, _ = bel(x, c0, seq, "reference")
        assert torch.allclose(yr, y, atol=5e-3)


@gpu
def test_prefix_uses_cached_beliefs_and_is_never_replayed(cfg, split, monkeypatch):
    """Chunked P1 bookkeeping (8-frame prefix, cached predicted beliefs, carried state, resets) gives the predicted
    beliefs of one contiguous scan; Mamba only ever scans the chunk's own 2T tokens."""
    cfg = copy.deepcopy(cfg)
    cfg["p1"]["precision"] = "fp32"
    torch.manual_seed(0)
    model = M.M0(cfg).cuda()
    eps = sorted(split["train"], key=lambda e: e.T)[:3]                 # short episodes: the chunks cross resets
    s = D.Streams(eps, 1, 8, seed=0)
    chunks = [s.next(20) for _ in range(3)]
    seen, lengths = [], []
    orig_mem, orig_fwd = model.belief.memory, model.belief.forward
    monkeypatch.setattr(model.belief, "memory", lambda b: (seen.append(b.detach().float()), orig_mem(b))[1])
    monkeypatch.setattr(model.belief, "forward", lambda x, *a, **k: (lengths.append(x.shape[1]), orig_fwd(x, *a, **k))[1])
    carry, cache = model.belief.initial(1, "cuda"), torch.zeros(1, 9, cfg["belief"]["width"], device="cuda")
    with torch.no_grad():
        for c in chunks:
            _, carry, cache = TR.p1_losses(model, c, carry, cache, None, cfg)
    assert lengths == [40, 40, 40]
    # contiguous reference: every emitted frame once, in order
    frames = {k: np.concatenate([c[k][:, 8:] for c in chunks], 1) for k in chunks[0]}
    t = TR.tensors(frames, "cuda")
    with torch.no_grad():
        Pc = model.perception(t["frames"], t["fixation"])
        x = model.belief.tokens(model.belief.percept_embed(Pc), t["reward"], t["terminated"], t["action"])
        y, _ = orig_fwd(x, model.belief.initial(1, "cuda"), t["first"].long().cumsum(1).repeat_interleave(2, 1))
        ref = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1) * (~t["first"])[..., None]
    for k in (1, 2):                                                    # chunk k: prefix + its 20 frames
        lo = 20 * k - 8
        assert torch.allclose(seen[k][0], ref[0, lo:lo + 28], atol=2e-3), k


# ---- denoiser ----

def _den_inputs(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    dims = [(M.N, cfg["perception"]["periphery"]["token_dim"]), (M.G, cfg["perception"]["fovea"]["token_dim"]),
            (cfg["perception"]["lighting"]["dim"],)]
    x = tuple(torch.randn(B, T, *d, generator=g) for d in dims)
    mem = torch.randn(B, T, cfg["denoiser"]["memory_tokens"], cfg["denoiser"]["width"], generator=g)
    sigma = torch.rand(B, T, generator=g) * 3 + 0.05
    act = torch.randint(0, 17, (B, T), generator=g)
    fix = torch.randint(1, 8, (B, T, 2), generator=g)
    return x, mem, sigma, act, fix, dims


def test_streaming_denoiser_equals_chunked(cfg):
    torch.manual_seed(0)
    den = M.Denoiser(cfg).eval()
    for blk in den.blocks:                                              # non-trivial adaLN so every path is used
        torch.nn.init.normal_(blk.mod.weight, std=0.02)
    for m in den.out:
        torch.nn.init.normal_(m.weight, std=0.02)
    B, T = 2, 11
    x, mem, sigma, act, fix, _ = _den_inputs(cfg, B, T)
    episode = torch.zeros(B, T, dtype=torch.long)
    episode[1, 6:] = 1                                                  # stream 1 resets at frame 6
    valid = torch.ones(B, T, dtype=torch.bool)
    with torch.no_grad():
        ref = den(x, sigma.log() / 4, act, mem, fix, episode, valid)
        for b in range(B):
            cache = den.new_cache()
            for t in range(T):
                if t and episode[b, t] != episode[b, t - 1]:
                    cache = den.new_cache()
                sl = lambda a: a[b:b + 1, t:t + 1]
                out, new = den.step(tuple(sl(a) for a in x), sl(sigma).log() / 4, sl(act), sl(mem), sl(fix), cache)
                cache = den.commit(cache, new)
                for o, r in zip(out, ref):
                    assert torch.allclose(o[0, 0], r[b, t], atol=1e-4), (b, t)


def test_cached_sampler_equals_recomputed_history(cfg):
    """The whole cached sampler (3 churned Euler evaluations, then one context pass per frame) equals a sampler that
    recomputes the full history with the stored context copies, under identical draws: several frames, an episode
    reset, branch duplication."""
    torch.manual_seed(0)
    den = M.Denoiser(cfg).eval()
    for blk in den.blocks:
        torch.nn.init.normal_(blk.mod.weight, std=0.02)
    for m in den.out:
        torch.nn.init.normal_(m.weight, std=0.02)
    e, s = cfg["edm"], cfg["edm"]["sampler"]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    B, T = 1, 6
    _, mem, _, act, fix, dims = _den_inputs(cfg, B, T)
    n = s["steps"]
    sig = [(s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
           for i in range(n)] + [0.0]
    gamma = min(s["s_churn"] / n, math.sqrt(2) - 1)

    def recompute(hist, t, gen, reps):
        """frame t given stored context copies hist [(p, g, v)] of frames since the last reset."""
        sl = lambda a: a[:, t - len(hist):t + 1].repeat_interleave(reps, 0)
        x = tuple(sig[0] * torch.randn((reps, 1, *d), generator=gen) for d in dims)
        for k in range(n):
            hat = sig[k] * (1 + gamma)
            x = tuple(a + math.sqrt(hat ** 2 - sig[k] ** 2) * torch.randn(a.shape, generator=gen) for a in x)
            seq = tuple(torch.cat([*(h[i] for h in hist), a], 1) if hist else a for i, a in enumerate(x))
            sg = torch.full((reps, len(hist) + 1), e["inference_context_sigma"])
            sg[:, -1] = hat
            c_skip, c_out, c_in, c_noise = M.precondition(sg)
            Fo = den(tuple(M._bcast(c_in, a) * a for a in seq), c_noise, sl(act), sl(mem), sl(fix),
                     torch.zeros(reps, len(hist) + 1, dtype=torch.long), torch.ones(reps, len(hist) + 1, dtype=torch.bool))
            Dn = tuple(M.out_map(M._bcast(c_skip[:, -1:], a) * a + M._bcast(c_out[:, -1:], f[:, -1:]) * f[:, -1:], clamp)
                       for a, f in zip(x, Fo))
            x = tuple(a + (sig[k + 1] - hat) * (a - d) / hat for a, d in zip(x, Dn))
        ctx = tuple(a + e["inference_context_sigma"] * torch.randn(a.shape, generator=gen) for a in x)
        return x, ctx

    with torch.no_grad():
        ga, gb = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
        cache, hist, reps = den.new_cache(), [], 1
        for t in range(T):
            if t == 3:                                                  # episode reset
                cache, hist = den.new_cache(), []
            if t == 4:                                                  # branch duplication
                cache = {"t": cache["t"], "kv": [(k.repeat_interleave(2, 0), v.repeat_interleave(2, 0)) for k, v in cache["kv"]]}
                hist, reps = [tuple(a.repeat_interleave(2, 0) for a in h) for h in hist], 2
            sl = lambda a: a[:, t:t + 1].repeat_interleave(reps, 0)
            xa = M.sample(den, dims, sl(act), sl(mem), sl(fix), cache, cfg, ga)
            cache = M.context_pass(den, xa, sl(act), sl(mem), sl(fix), cache, cfg, ga)
            xb, ctx = recompute(hist, t, gb, reps)
            hist = (hist + [ctx])[-den.window:]
            for u, v in zip(xa, xb):
                assert torch.allclose(u, v, atol=1e-4), t


def test_edm_coefficients_loss_and_sampler_equations(cfg, monkeypatch):
    sd, sigma = 1.0, torch.tensor([0.002, 0.1, 1.0, 20.0])
    c_skip, c_out, c_in, c_noise = M.precondition(sigma, sd)
    assert torch.allclose(c_skip, sd ** 2 / (sigma ** 2 + sd ** 2))
    assert torch.allclose(c_out, sigma * sd / (sigma ** 2 + sd ** 2).sqrt())
    assert torch.allclose(c_in, 1 / (sigma ** 2 + sd ** 2).sqrt()) and torch.allclose(c_noise, sigma.log() / 4)
    # raw-output MSE = clean-space MSE weighted by 1 / c_out^2 (= 1 + 1 / sigma^2 at sd = 1)
    g = torch.Generator().manual_seed(0)
    x0, n, Fo = (torch.randn(4, 10, generator=g, dtype=torch.float64) for _ in range(3))
    s = sigma.double()[:, None]
    cs, co, _, _ = M.precondition(s, sd)
    x = x0 + s * n
    raw = (Fo - (x0 - cs * x) / co) ** 2
    assert torch.allclose(raw, (cs * x + co * Fo - x0) ** 2 * (1 + 1 / s ** 2), rtol=1e-9)
    # the sampler follows EDM Algorithm 2 (NVIDIA generate.py, order 1): a stub denoiser with D = 0.5 x
    monkeypatch.setattr(M, "out_map", lambda a, c: a)

    class Stub:
        window = 4

        def step(self, xc, c_noise, *a):
            sig = (4 * c_noise).exp()
            cs, co, ci, _ = M.precondition(sig)
            return tuple((0.5 * t / M._bcast(ci, t) - M._bcast(cs, t) * t / M._bcast(ci, t)) / M._bcast(co, t)
                         for t in xc), []

    e = cfg["edm"]["sampler"]
    ga, gb = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    out = M.sample(Stub(), [(5,)], None, torch.zeros(2, 1, 1), None, None, cfg, ga)[0]
    N = e["steps"]
    steps = [(e["sigma_max"] ** (1 / e["rho"]) + i / (N - 1) * (e["sigma_min"] ** (1 / e["rho"]) - e["sigma_max"] ** (1 / e["rho"]))) ** e["rho"]
             for i in range(N)] + [0.0]
    x = torch.randn(2, 1, 5, generator=gb) * steps[0]
    for i in range(N):
        gam = min(e["s_churn"] / N, math.sqrt(2) - 1)
        t_hat = steps[i] + gam * steps[i]
        x_hat = x + math.sqrt(t_hat ** 2 - steps[i] ** 2) * e["s_noise"] * torch.randn(x.shape, generator=gb)
        d = (x_hat - 0.5 * x_hat) / t_hat
        x = x_hat + (steps[i + 1] - t_hat) * d
    assert torch.allclose(out, x, atol=1e-6)


# ---- outcome heads ----

def test_reward_decoding_and_bin_probabilities(cfg):
    torch.manual_seed(0)
    out = M.Outcome(cfg)
    rewards = torch.tensor([0.0] * 50 + [1.0] * 5 + [0.1] * 3 + [-0.1] * 3 + [0.9, 1.1, -0.9])
    k = out.bin_of(rewards)
    for b in k.unique():
        out.bin_value[b] = rewards[k == b].mean()
    assert len(k.unique()) == len(rewards.unique())                     # every observed value has its own bin
    H = out.reward.in_features
    for r in (0.0, 1.0):                                                # a head with all mass on one bin samples it
        with torch.no_grad():
            out.reward.weight.zero_()
            out.reward.bias.zero_()
            out.reward.bias[out.bin_of(torch.tensor(r))] = 30.0
        s = out.sample_reward(torch.zeros(1000, H))
        assert torch.all(s == r), r
    with torch.no_grad():                                               # a broad head: frequencies = exp(log P)
        torch.nn.init.normal_(out.reward.weight, std=0.05)
        torch.nn.init.normal_(out.reward.bias, std=1.0)
    h = torch.randn(1, H)
    lp = out.bin_logp(h)[0].detach()
    assert abs(float(lp.exp().sum()) - 1) < 1e-4
    g = torch.Generator().manual_seed(0)
    s = out.sample_reward(h.expand(200_000, -1), g)
    freq = torch.bincount(out.bin_of(s), minlength=out.bins).float() / len(s)
    assert float((freq - lp.exp()).abs().max()) < 0.005                # each decoded value lies in its own bin
    vals = torch.tensor([0.0, 1.0, 0.1])
    assert torch.allclose(out.reward_logp(h.expand(3, -1), vals), lp[out.bin_of(vals)], atol=1e-5)


# ---- gradient routes, per loss ----

@gpu
def test_gradient_routes_per_loss(cfg, split, monkeypatch):
    torch.manual_seed(0)
    model = M.M0(cfg).cuda()
    for blk in model.denoiser.blocks:
        torch.nn.init.normal_(blk.mod.weight, std=0.02)
    for m in model.denoiser.out:
        torch.nn.init.normal_(m.weight, std=0.02)
    eps = split["train"][:4]
    s = D.Streams(eps, 2, 8, seed=0, targets_offsets=D.frame_offsets(eps))
    s.next(8)
    chunk = s.next(16)
    targets = np.random.default_rng(0).random((int(D.frame_offsets(eps)[-1]), 7, 9)).astype(np.float16)
    W = cfg["belief"]["width"]

    def grads(term, detach_memory=False):
        model.zero_grad(set_to_none=True)
        if detach_memory:
            orig = model.belief.memory
            monkeypatch.setattr(model.belief, "memory", lambda b: orig(b).detach())
        terms, _, _ = TR.p1_losses(model, chunk, model.belief.initial(2, "cuda"), torch.zeros(2, 9, W, device="cuda"),
                                   targets, cfg)
        terms[term].backward()
        if detach_memory:
            monkeypatch.undo()
        norm = lambda mod: sum(float(p.grad.norm()) for p in mod.parameters() if p.grad is not None)
        return {k: norm(getattr(model, k)) for k in ("perception", "belief", "denoiser", "tsp", "pixels", "change", "outcome")}

    g = grads("g1")
    assert g["perception"] > 0 and g["belief"] > 0 and g["denoiser"] > 0
    g = grads("g1", detach_memory=True)                                 # only the memory tokens reach the encoder
    assert g["perception"] == 0 and g["belief"] == 0 and g["denoiser"] > 0
    g = grads("tsp")
    assert g["perception"] > 0 and g["belief"] > 0 and g["tsp"] > 0 and g["denoiser"] == 0
    g = grads("rec")
    assert g["perception"] > 0 and g["belief"] == 0 and g["denoiser"] == 0
    fovea = sum(float(p.grad.norm()) for p in model.perception.fovea.parameters() if p.grad is not None)
    assert fovea > 0
    g = grads("change")
    assert g["perception"] > 0 and g["belief"] > 0 and g["change"] > 0 and g["denoiser"] == 0
    for term in ("reward", "cont"):
        g = grads(term)
        assert g["perception"] > 0 and g["belief"] > 0 and g["outcome"] > 0 and g["denoiser"] == 0


# ---- resume ----

@gpu
def test_resumed_run_equals_uninterrupted(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["p1"]["frames"] = 4 * 256
    c["p1"]["monitor"]["windows"] = 4
    c["optimizer"]["warmup_frames"] = 256
    c["data"]["streams"] = 4
    a = TR.p1(c, 3, tmp_path / "a", updates=4, n_episodes=20)
    TR.p1(c, 3, tmp_path / "b", updates=2, n_episodes=20)
    b = TR.p1(c, 3, tmp_path / "b", updates=4, n_episodes=20)
    diff = max(float((u - v).abs().max()) for u, v in zip(a.state_dict().values(), b.state_dict().values())
               if u.is_floating_point())
    assert diff < 1e-3, diff
