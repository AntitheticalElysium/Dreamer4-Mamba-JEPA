"""Pre-launch mechanical checks (OVERNIGHT_PROPOSAL.md's contract), GPU, one process, 2 streams of P1-F's guided data:
1. Disabled path: HEAD's p1_losses (a saved copy, argv[2]) vs the edited one without the key: every term, every
   parameter gradient and the parameters after one AdamW step at a nonzero LR (clipped as p1 clips) bitwise equal, at
   P1-F's initialization and at its checkpoint 5 (with its Adam moments).
2. Enabled path vs Codex's BM arm (beta_fork.losses on the unchanged config): the same, bitwise.
3. Routing: the parameters the TSP term alone reaches, faithful vs scale-weighted: the same set, none in the pixel
   decoder's scale network; finite.
4. Resume across the teacher identity, with TR.p1 itself at a shrunken frame budget (checkpoint every 2 updates):
   4 straight updates vs 2 + resume + 2, against a second straight run (cross-process noise floor); a scale-weighted
   checkpoint is refused by validate_resume without the key, P1-F's with it.
Usage (project venv, GPU): python -m artifacts.experiments.20261101_beta_tsp.checks OUT_JSON HEAD_TRAIN_PY SCRATCH_DIR [--resume]
  (run without, then with --resume)
"""
import copy, functools, importlib.util, json, os, shutil, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR

TG = importlib.import_module("artifacts.experiments.20261030_p1_guided.train_guided")
BM = importlib.import_module("artifacts.experiments.20261031_grounding_loss_review.followup.beta_fork")
P1F = TR.ROOT / "artifacts/experiments/20261030_p1_guided/run"
DEV, KEY = torch.device("cuda"), "tsp_mean_loss"


def start(cfg, ck):
    """Model, optimizer, 2-stream chunk, carry and cache: P1-F's checkpoint `ck`, or p1's initialization if None."""
    split = D.load_split(cfg)
    torch.manual_seed(TG.WORLD_SEED)
    streams = TG.FT2.GuidedStreams(split["train"], cfg["data"]["streams"], cfg["data"]["context_prefix"], seed=TG.WORLD_SEED,
                                   targets_offsets=D.frame_offsets(split["train"]), place=np.load(TG.M05.PLACE))
    model = M.M0(cfg).to(DEV); model.train(); model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    TR.set_reward_values(model, split["train"])
    o = cfg["optimizer"]
    opt = torch.optim.AdamW(model.param_groups(), lr=o["lr"], betas=tuple(o["betas"]), eps=o["eps"])
    rows = np.arange(2)
    if ck is None:
        carry = M.carry_rows(model.belief.initial(cfg["data"]["streams"], DEV), rows)
        cache = torch.zeros(2, cfg["data"]["context_prefix"] + 1, cfg["belief"]["width"], device=DEV)
    else:
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); streams.load_state(ck["streams"])
        carry, cache = M.carry_rows(ck["carry"], rows), ck["cache"][:2]
    return model, opt, D.select(streams.next(64), rows), carry, cache, {n: p.detach().clone() for n, p in model.named_parameters()}, \
        copy.deepcopy(opt.state_dict())


def branch(model, opt, init, opt0, fn, chunk, carry, cache, targets, cfg, rng):
    """Restore parameters and optimizer, compute fn's loss at the saved RNG, backward, clip, one step at a nonzero LR."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            p.copy_(init[n])
    opt.load_state_dict(copy.deepcopy(opt0))                              # load_state_dict aliases same-device tensors
    for g in opt.param_groups:
        g["lr"] = cfg["optimizer"]["lr"] * g["lr_scale"] * 0.5
    opt.zero_grad(set_to_none=True)
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    terms = fn(model, chunk, M.carry_rows(carry, np.arange(2)), cache.clone(), targets, cfg)[0]
    loss = model.harmoniser(TR.groups(terms)) + terms["noise"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optimizer"]["grad_clip"])
    opt.step()
    return {k: float(v.detach()) for k, v in terms.items()}, grads, {n: p.detach().clone() for n, p in model.named_parameters()}


def same(a, b):
    ta, ga, pa = a; tb, gb, pb = b
    return {"terms_equal": ta == tb, "grad_keys_equal": ga.keys() == gb.keys(),
            "grad_max_abs": max(float((ga[n] - gb[n]).abs().max()) for n in ga) if ga.keys() == gb.keys() else None,
            "param_after_step_max_abs": max(float((pa[n] - pb[n]).abs().max()) for n in pa), "terms": [ta, tb]}


def tsp_routes(model, fn, chunk, carry, cache, targets, cfg, rng):
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    tsp = fn(model, chunk, M.carry_rows(carry, np.arange(2)), cache.clone(), targets, cfg)[0]["tsp"]
    g = torch.autograd.grad(tsp, [p for p in model.parameters()], allow_unused=True)
    names = [n for n, _ in model.named_parameters()]
    return float(tsp.detach()), sorted({n.rsplit(".", 1)[0] for n, x in zip(names, g) if x is not None and float(x.abs().max()) > 0})


def resume_test(cfg, out):
    """TR.p1 at a shrunken budget (cfg's frames 20,480: a checkpoint every 2 updates) under P1-F's guided streams."""
    small = copy.deepcopy(cfg); small["p1"]["frames"] = 20480
    D.Streams = functools.partial(TG.FT2.GuidedStreams, place=np.load(TG.M05.PLACE))
    res = {}
    for name, plan in (("straight_a", (4,)), ("straight_b", (4,)), ("resumed", (2, 4))):
        d = out / name
        shutil.rmtree(d, ignore_errors=True)
        for u in plan:
            TR.p1(small, TG.WORLD_SEED, d, u, None, None, True)
        res[name] = torch.load(d / "ckpt_02.pt", weights_only=False, map_location="cpu")
    diff = lambda a, b: max(float((res[a]["model"][k].float() - res[b]["model"][k].float()).abs().max()) for k in res[a]["model"])
    out_ = {"straight_a_vs_b_max_abs": diff("straight_a", "straight_b"), "straight_a_vs_resumed_max_abs": diff("straight_a", "resumed"),
            "state_equal": res["straight_a"]["state"] == res["resumed"]["state"]}
    plain = copy.deepcopy(small); plain["heads"]["pixel_decoder"].pop(KEY)
    prov = TR.run_provenance(small)
    for name, (ck, c) in {"refuses_without_key": (res["resumed"], plain),
                          "refuses_p1f_with_key": (torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu"), cfg)}.items():
        try:
            TR.validate_resume(ck, c, TG.WORLD_SEED, prov); out_[name] = False
        except ValueError as e:
            out_[name] = f"refused: {e}"
    return out_


def main():
    out, head = Path(sys.argv[1]), sys.argv[2]
    spec = importlib.util.spec_from_file_location("train_head", head)
    HEAD = importlib.util.module_from_spec(spec); spec.loader.exec_module(HEAD)
    if "--resume" in sys.argv:                                            # its own process (6 GB card)
        cfg = torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu")["cfg"]
        on = copy.deepcopy(cfg); on["heads"]["pixel_decoder"][KEY] = "scale_weighted_nll"
        res = json.loads(out.read_text())
        res["resume"] = resume_test(on, Path(sys.argv[3]))
        print("resume", json.dumps(res["resume"]), flush=True)
        out.write_text(json.dumps(res, indent=1))
        return
    res = {"head_train_py": head, "head_sha256": TR.file_sha256(head), "edited_sha256": TR.file_sha256(TR.ROOT / "saccade/train.py")}
    for at in ("init", "ckpt_05"):
        ck = None if at == "init" else torch.load(P1F / "ckpt_05.pt", weights_only=False, map_location=DEV)
        cfg = (ck or torch.load(P1F / "ckpt_01.pt", weights_only=False, map_location="cpu"))["cfg"]
        assert KEY not in cfg["heads"]["pixel_decoder"] and cfg["heads"]["pixel_decoder"]["mean_loss"] == "faithful"
        on = copy.deepcopy(cfg); on["heads"]["pixel_decoder"][KEY] = "scale_weighted_nll"
        targets = np.load(TR.ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
        model, opt, chunk, carry, cache, init, opt0 = start(cfg, ck)
        rng = [torch.get_rng_state(), torch.cuda.get_rng_state()]
        args = (chunk, carry, cache, targets)
        bm = lambda m, ch, ca, cc, t, c: BM.losses(m, ch, ca, cc, t, cfg, "BM")
        r = {"disabled_vs_head": same(branch(model, opt, init, opt0, HEAD.p1_losses, *args, cfg, rng),
                                      branch(model, opt, init, opt0, TR.p1_losses, *args, cfg, rng)),
             "enabled_vs_codex_bm": same(branch(model, opt, init, opt0, bm, *args, cfg, rng),
                                         branch(model, opt, init, opt0, TR.p1_losses, *args, on, rng))}
        for n, (fn, c) in {"faithful": (TR.p1_losses, cfg), "scale_weighted": (TR.p1_losses, on)}.items():
            v, mods = tsp_routes(model, fn, *args, c, rng)
            r[f"tsp_{n}"] = {"value": v, "modules_reached": mods}
        r["routes_equal"] = r["tsp_faithful"]["modules_reached"] == r["tsp_scale_weighted"]["modules_reached"]
        r["scale_network_reached"] = [m for m in r["tsp_scale_weighted"]["modules_reached"] if m.startswith(("pixels.scale", "pixels.cell"))]
        res[at] = r
        print(at, json.dumps({k: (v if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "terms"}) for k, v in r.items()
                              if k not in ("tsp_faithful", "tsp_scale_weighted")}), flush=True)
        del model, opt, ck
        torch.cuda.empty_cache()
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
