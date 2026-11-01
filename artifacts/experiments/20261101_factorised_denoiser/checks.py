"""Decision 1's GPU build checks (PLAN.md 1-6), one step per process on the 6 GB card; results merge into checks.json.
Usage (project venv, GPU): python -m artifacts.experiments.20261101_factorised_denoiser.checks STEP PRE_DIR SCRATCH
  STEP: disabled | launcher | invariance | cost | cache. PRE_DIR holds the code compared against as model.py / train.py:
  60b98881's (before decision 1) for disabled, a4f2c2a2's (before the stage-2 cache) for cache; SCRATCH receives the
  launcher's 25-update run.
"""
import copy, importlib, importlib.util, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR

BC = importlib.import_module("artifacts.experiments.20261101_beta_tsp.checks")   # start / branch / same (its checks 1)
TG, P1F, DEV = BC.TG, BC.P1F, BC.DEV
HERE = Path(__file__).parent
CEN = torch.stack(torch.meshgrid(torch.arange(1, 8), torch.arange(1, 8), indexing="ij"), -1).reshape(-1, 2)   # CPU: the launcher step
# leaves the GPU to its child (a parent CUDA context plus autotuning ran it out of memory on the 6 GB card)


def factorised(cfg):
    """The config as a launcher switches decision 1 on (m0.yaml's documented keys)."""
    c = copy.deepcopy(cfg)
    c["denoiser"]["factorised"], c["edm"]["sampler"]["glimpse_conditioning_sigma"] = True, 0.1
    return c


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def targets_of(cfg):
    return np.load(TR.ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")


def disabled(pre):
    """1: the pre-decision code (its p1_losses bound to its own model module) against HEAD without the key."""
    PM, PT = load(pre / "model.py", "pre_model"), load(pre / "train.py", "pre_train")
    PT.M = PM
    cfg0 = torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu")["cfg"]
    assert "factorised" not in cfg0["denoiser"]
    torch.manual_seed(TG.WORLD_SEED); a = M.M0(cfg0)
    torch.manual_seed(TG.WORLD_SEED); b = PM.M0(cfg0)
    res = {"init_params_equal": all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values()))
           and a.state_dict().keys() == b.state_dict().keys()}
    del a, b
    for at in ("init", "ckpt_05"):
        ck = None if at == "init" else torch.load(P1F / "ckpt_05.pt", weights_only=False, map_location=DEV)
        cfg = ck["cfg"] if ck else cfg0
        model, opt, chunk, carry, cache, init, opt0 = BC.start(cfg, ck)
        old = PM.M0(cfg).to(DEV); old.train(); old.denoiser.checkpoint = model.denoiser.checkpoint
        old.load_state_dict(model.state_dict())
        o = cfg["optimizer"]
        oopt = torch.optim.AdamW(old.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
        if ck:
            oopt.load_state_dict(ck["opt"])
        del ck
        oinit, oopt0 = {n: p.detach().clone() for n, p in old.named_parameters()}, copy.deepcopy(oopt.state_dict())
        rng = [torch.get_rng_state(), torch.cuda.get_rng_state()]
        args = (chunk, carry, cache, targets_of(cfg), cfg, rng)
        r = BC.same(BC.branch(old, oopt, oinit, oopt0, PT.p1_losses, *args), BC.branch(model, opt, init, opt0, TR.p1_losses, *args))
        res[at] = r
        print(at, json.dumps({k: v for k, v in r.items() if k != "terms"}), flush=True)
        del model, opt, old, oopt
        torch.cuda.empty_cache()
    return res


def launcher(scratch):
    """2: P1-F's own launcher for 25 updates against P1-F's logged updates 1 and 25."""
    out = scratch / "launcher25"
    shutil.rmtree(out, ignore_errors=True)
    subprocess.run([sys.executable, "-m", "artifacts.experiments.20261030_p1_guided.train_guided", str(out), "--updates", "25",
                    "--no-checkpoints"], check=True, cwd=TR.ROOT)
    ref = [json.loads(l) for l in open(P1F / "log.jsonl")][:2]
    new = [json.loads(l) for l in open(out / "log.jsonl")]
    skip = ("sec_per_update", "peak_gib")
    res = {f"update{r['update']}_differs": {k: [r[k], n.get(k)] for k in r if k not in skip and r[k] != n.get(k)}
           for r, n in zip(ref, new)}
    res["updates_compared"] = [n["update"] for n in new]
    res["timing_p1f_vs_now"] = [[r["sec_per_update"], n["sec_per_update"], r["peak_gib"], n["peak_gib"]] for r, n in zip(ref, new)]
    return res


def capture(model, chunk, carry, cache, cfg):
    """The denoiser's real inputs as p1_losses builds them (its denoise_loss call, positional)."""
    rec, orig = {}, M.denoise_loss
    def spy(*a, **k):
        rec["a"] = a
        return orig(*a, **k)
    M.denoise_loss = spy
    try:
        with torch.no_grad():
            TR.p1_losses(model, chunk, M.carry_rows(carry, np.arange(2)), cache.clone(), targets_of(cfg), cfg)
    finally:
        M.denoise_loss = orig
    return rec["a"]


def invariance():
    """3 and 4 on P1-F's checkpoint-10 weights switched to factorised; 5 at P1-F's initialization."""
    ck = torch.load(P1F / "ckpt_10.pt", weights_only=False, map_location=DEV)
    cfg = factorised(ck["cfg"])
    model, opt, chunk, carry, cache, init, opt0 = BC.start(cfg, ck)
    del ck, opt
    model.eval()
    den, x0, sigma, act, mem, fix, episode, valid, sd = capture(model, chunk, carry, cache, cfg)
    CEN = globals()["CEN"].to(DEV)
    assert isinstance(sigma, tuple)
    gen = torch.Generator(device=DEV).manual_seed(0)
    pre = [M.precondition(s, sd) for s in M._parts(sigma)]
    xs = tuple(t + M._bcast(s, t) * torch.randn(t.shape, device=DEV, generator=gen) for t, s in zip(x0, M._parts(sigma)))
    xin, cn = tuple(M._bcast(c[2], t) * t for t, c in zip(xs, pre)), (pre[0][3], pre[1][3])
    res = {"frames": list(fix.shape[:2]), "fixations_distinct_in_window": int(torch.unique(fix.reshape(-1, 2), dim=0).shape[0])}
    with torch.no_grad(), TR.autocast(DEV, cfg):
        run = lambda x, c, f: den(x, c, act, mem, f, episode, valid)
        ref = run(xin, cn, fix)
        same = {"p": 0, "v": 0, "g": 0}
        for w in range(49):
            o = run(xin, cn, CEN[w].expand_as(fix))
            for k, i in (("p", 0), ("g", 1), ("v", 2)):
                same[k] += int(torch.equal(o[i], ref[i]))
        res["fixation_sweep_equal_of_49"] = same
        xg = (xin[0], torch.randn(xin[1].shape, device=DEV, generator=gen), xin[2])
        o = run(xg, (cn[0], torch.full_like(cn[1], math.log(2.0) / 4)), fix)
        res["glimpse_inputs_and_level_replaced_equal"] = {"p": torch.equal(o[0], ref[0]), "v": torch.equal(o[2], ref[2]),
                                                          "g": torch.equal(o[1], ref[1])}
        res["output_scale"] = [float(t.float().abs().mean()) for t in ref]
        cache_s = den.new_cache()
        for t in range(8):
            sl = lambda a: a[:, t:t + 1]
            cache_s = M.context_pass(den, tuple(sl(p) for p in x0), sl(act), sl(mem), sl(fix), cache_s, cfg, gen)
        shapes, sl = [tuple(p.shape[2:]) for p in x0], lambda a: a[:, 8:9]
        draws = [M.sample(den, shapes, sl(act), sl(mem), CEN[w].expand_as(sl(fix)), cache_s, cfg, torch.Generator(device=DEV).manual_seed(5))
                 for w in range(49)]
        res["cascade_equal_to_w0_of_49"] = {k: sum(int(torch.equal(d[i], draws[0][i])) for d in draws) for k, i in (("p", 0), ("g", 1), ("v", 2))}
        hook = M.sample(den, shapes, sl(act), sl(mem), sl(fix), cache_s, cfg, torch.Generator(device=DEV).manual_seed(5),
                        choose=lambda p, v: CEN[7].expand_as(sl(fix)))
        res["choose_hook"] = {"fix_returned": torch.equal(hook[1], CEN[7].expand_as(sl(fix))),
                              "frame_equals_draw_at_it": all(torch.equal(a, b) for a, b in zip(hook[0], draws[7]))}
    del model
    torch.cuda.empty_cache()
    cfg0 = torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu")["cfg"]
    model, opt, chunk, carry, cache, init, opt0 = BC.start(factorised(cfg0), None)
    rng = [torch.get_rng_state(), torch.cuda.get_rng_state()]
    args = (chunk, carry, cache, targets_of(cfg0))
    tf, gf, _ = BC.branch(model, opt, init, opt0, TR.p1_losses, *args, factorised(cfg0), rng)
    model.denoiser.factorised = False
    tj, gj, _ = BC.branch(model, opt, init, opt0, TR.p1_losses, *args, cfg0, rng)
    res["training_path"] = {"terms_finite": all(math.isfinite(v) for v in tf.values()), "grad_keys_equal": gf.keys() == gj.keys(),
                            "terms_factorised": tf, "terms_joint": tj}
    return res


def update_cost(cfg, S, n=4):
    """Median seconds of a real P1 update (TR.p1's loop body: microbatches, backward, clip, step) and peak memory."""
    split = D.load_split(cfg)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    torch.manual_seed(TG.WORLD_SEED)
    streams = TG.FT2.GuidedStreams(split["train"], d["streams"], P, seed=TG.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]),
                                   place=np.load(TG.M05.PLACE))
    model = M.M0(cfg).to(DEV); model.train(); model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    TR.set_reward_values(model, split["train"])
    o, targets = cfg["optimizer"], targets_of(cfg)
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    carry, cache = model.belief.initial(d["streams"], DEV), torch.zeros(d["streams"], P + 1, cfg["belief"]["width"], device=DEV)
    micro = 1 if S == d["chunk_steps"][0] else d["accumulate_at_256"]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n + 1):
        torch.cuda.synchronize(); t0 = time.time()
        chunk = streams.next(S)
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), micro):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], targets, cfg)
            ((model.harmoniser(TR.groups(terms)) + terms.get("noise", 0)) / micro).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
        del terms, c_new, cache_new
        torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
        opt.step()
        torch.cuda.synchronize(); times.append(time.time() - t0)
    out = {"sec_per_update": float(np.median(times[1:])), "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    del model, opt, carry, cache
    torch.cuda.empty_cache()
    return out


def sample_cost(cfg, B=64, n=10):
    """M.sample per call at B streams after an 8-frame context, on P1-F's checkpoint-10 weights, joint vs cascade."""
    ck = torch.load(P1F / "ckpt_10.pt", weights_only=False, map_location="cpu")
    model = M.M0(cfg).to(DEV).eval(); model.load_state_dict(ck["model"])
    den, gen = model.denoiser, torch.Generator(device=DEV).manual_seed(0)
    dims = [(M.N, cfg["perception"]["periphery"]["token_dim"]), (M.G, cfg["perception"]["fovea"]["token_dim"]), (cfg["perception"]["lighting"]["dim"],)]
    rnd = lambda *s: torch.randn(*s, device=DEV, generator=gen)
    act, mem = torch.zeros(B, 1, dtype=torch.long, device=DEV), rnd(B, 1, cfg["denoiser"]["memory_tokens"], cfg["denoiser"]["width"])
    fix = CEN[24].to(DEV).expand(B, 1, 2)
    out = {}
    with torch.no_grad(), TR.autocast(DEV, cfg):
        cache = den.new_cache()
        for _ in range(8):
            cache = M.context_pass(den, tuple(rnd(B, 1, *s) for s in dims), act, mem, fix, cache, cfg, gen)
        for name, flag in (("joint", False), ("cascade", True)):
            den.factorised = flag
            for i in range(n + 2):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                M.sample(den, dims, act, mem, fix, cache, cfg, gen)
            torch.cuda.synchronize(); out[f"{name}_ms_per_call"] = (time.time() - t0) / n * 1000
    out["ratio"] = out["cascade_ms_per_call"] / out["joint_ms_per_call"]
    return out


def cost():
    """6: from the same initialization, joint (P1-F's recipe) vs factorised."""
    cfg0 = torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu")["cfg"]
    res = {}
    for name, c in (("joint", cfg0), ("factorised", factorised(cfg0))):
        for S in cfg0["data"]["chunk_steps"]:
            res[f"{name}_update_{S}"] = update_cost(c, S)
            print(name, S, res[f"{name}_update_{S}"], flush=True)
    res["sample"] = sample_cost(factorised(cfg0))
    return res


def cache(pre):
    """7 (PLAN, added with the stage-2 cache): against the code before it, the factorised training step bitwise (P1-F's
    initialization, 2 streams: terms, gradients, one step), and on P1-F's checkpoint-10 weights switched on, the
    cascade's frames under common draws (periphery and lighting bitwise; the glimpse within float noise) and the cost
    per sampled frame at 64 streams after an 8-frame context."""
    PM, PT = load(pre / "model.py", "pre_cache_model"), load(pre / "train.py", "pre_cache_train")
    PT.M = PM
    cfg = factorised(torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu")["cfg"])
    model, opt, chunk, carry, cache_, init, opt0 = BC.start(cfg, None)
    old = PM.M0(cfg).to(DEV); old.train(); old.denoiser.checkpoint = model.denoiser.checkpoint
    old.load_state_dict(model.state_dict())
    o = cfg["optimizer"]
    oopt = torch.optim.AdamW(old.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    oinit, oopt0 = {n: p.detach().clone() for n, p in old.named_parameters()}, copy.deepcopy(oopt.state_dict())
    rng = [torch.get_rng_state(), torch.cuda.get_rng_state()]
    args = (chunk, carry, cache_, targets_of(cfg), cfg, rng)
    res = {"training_factorised": BC.same(BC.branch(old, oopt, oinit, oopt0, PT.p1_losses, *args),
                                          BC.branch(model, opt, init, opt0, TR.p1_losses, *args))}
    res["training_factorised"].pop("terms")
    del model, opt, old, oopt
    torch.cuda.empty_cache()
    ck = torch.load(P1F / "ckpt_10.pt", weights_only=False, map_location="cpu")
    dens = []
    for mod in (M, PM):
        m = mod.M0(cfg).to(DEV).eval(); m.load_state_dict(ck["model"]); dens.append((mod, m.denoiser))
    B, gen = 64, torch.Generator(device=DEV).manual_seed(0)
    dims = [(M.N, cfg["perception"]["periphery"]["token_dim"]), (M.G, cfg["perception"]["fovea"]["token_dim"]), (cfg["perception"]["lighting"]["dim"],)]
    rnd = lambda *s: torch.randn(*s, device=DEV, generator=gen)
    act, mem, fix = torch.randint(0, 17, (B, 1), device=DEV, generator=gen), rnd(B, 1, cfg["denoiser"]["memory_tokens"], cfg["denoiser"]["width"]), \
        CEN.to(DEV)[torch.randint(0, 49, (B,), device=DEV, generator=gen)][:, None]
    ctx = [tuple(rnd(B, 1, *s) for s in dims) for _ in range(8)]
    with torch.no_grad(), TR.autocast(DEV, cfg):
        caches = []
        for mod, den in dens:
            c_ = den.new_cache()
            for t in range(8):
                c_ = mod.context_pass(den, ctx[t], act, mem, fix, c_, cfg, torch.Generator(device=DEV).manual_seed(100 + t))
            caches.append(c_)
        res["context_caches_equal"] = all(torch.equal(a, b) for (ka, va), (kb, vb) in zip(caches[0]["kv"], caches[1]["kv"]) for a, b in ((ka, kb), (va, vb)))
        cmp = []
        for seed in range(4):
            a, b = (mod.sample(den, dims, act, mem, fix, c_, cfg, torch.Generator(device=DEV).manual_seed(seed))
                    for (mod, den), c_ in zip(dens, caches))
            cmp.append({"p_equal": torch.equal(a[0], b[0]), "v_equal": torch.equal(a[2], b[2]),
                        "g_max_abs": float((a[1] - b[1]).abs().max()), "g_std": float(b[1].float().std())})
        res["frames_cached_vs_uncached"] = cmp
        for name, (mod, den), c_, flag in (("joint", dens[0], caches[0], False), ("cascade_cached", dens[0], caches[0], True),
                                           ("cascade_uncached", dens[1], caches[1], True)):
            den.factorised = flag
            for i in range(12):
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                mod.sample(den, dims, act, mem, fix, c_, cfg, gen)
            torch.cuda.synchronize(); res[f"{name}_ms_per_frame"] = (time.time() - t0) / 10 * 1000
            den.factorised = True
    return res


def main():
    step, pre, scratch = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    f = HERE / "checks.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    step_key = step + ("" if os.environ.get("MAMBA_DETERMINISTIC") == "1" else "_autotuned")   # mamba_ssm's mode
    res[step_key] = {"disabled": lambda: disabled(pre), "launcher": lambda: launcher(scratch), "invariance": invariance, "cost": cost,
                     "cache": lambda: cache(pre)}[step]()
    res.setdefault("code", {})[step_key] = {**TR.run_provenance(TR.load_cfg()), "pre_sha256": {
        f: TR.file_sha256(pre / f) for f in ("model.py", "train.py")}}
    print(step_key, json.dumps(res[step_key], default=str)[:3000], flush=True)
    f.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
