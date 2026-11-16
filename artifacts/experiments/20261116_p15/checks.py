"""P1.5 pre-launch checks (PLAN.md), GPU:

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.checks mamba3      the Mamba-3 wrapper
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.checks units       record, routing, harmoniser, gaze
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.checks throughput  run-scale cost, route on / off
"""
import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from saccade import chooser as C, data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
P7 = importlib.import_module("artifacts.experiments.20261116_p15.train")
DEV = torch.device("cuda")
OUT = {}


def cfg_of(route=True):
    assert P7.READ in ("linear", "mlp"), "set from test 3's verdict first"
    return P7.recipe(route)


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-12))


def mamba3():
    """The wrapper against the official module, chunk carry, episode resets, several streams, gradients."""
    torch.manual_seed(0)
    b = {**P7.F61.recipe()["belief"], "mixer": "mamba3", "mamba3": {"chunk_size": 64}}   # P7.recipe's belief section
    w = M.FunctionalMamba3(b).to(DEV)
    L, res = 128, {}
    u = torch.randn(3, L, b["width"], device=DEV)
    for name, ac in (("fp32", False), ("bf16", True)):
        with torch.autocast("cuda", torch.bfloat16, enabled=ac):
            ref = w.core(u[:1])                                          # the official forward, one fresh sequence
            y, c = w(u[:1], w.initial(1))
            y1, c1 = w(u[:1, :L // 2], w.initial(1))
            y2, c2 = w(u[:1, L // 2:], c1, torch.zeros(1, L // 2, dtype=torch.long, device=DEV))
            seq = torch.zeros(1, L, dtype=torch.long, device=DEV); seq[:, 40:] = 1
            yr, _ = w(u[:1], w.initial(1), seq)
            fresh, _ = w(u[:1, 40:], w.initial(1))
            ya, _ = w(u[:1, :40], w.initial(1))
            res[name] = {"official_vs_wrapper": rel(y, ref), "chunked_vs_contiguous": rel(torch.cat([y1, y2], 1), y),
                         "carry_ssm": rel(c2.ssm, c.ssm), "carry_angle": rel(c2.angle, c.angle),
                         "reset_after": rel(yr[:, 40:], fresh), "reset_before": rel(yr[:, :40], ya)}
            starts = [None, 17, 90]                                      # three streams, a carried state, mixed resets
            seqs = torch.zeros(3, L, dtype=torch.long, device=DEV)
            for i, s in enumerate(starts):
                if s is not None:
                    seqs[i, s:] = 1
            _, warm = w(u, w.initial(3))
            yb, cb = w(u.flip(1), warm, seqs)
            per = []
            for i, s in enumerate(starts):
                one = M.Mamba3Carry(*(getattr(warm, f)[i:i + 1] for f in ("angle", "ssm", "k", "v")))
                yi, ci = w(u.flip(1)[i:i + 1], one, seqs[i:i + 1])
                per.append(rel(yb[i:i + 1], yi))
                per.append(rel(cb.ssm[i:i + 1], ci.ssm))
            res[name]["streams_vs_alone"] = max(per)
    y, _ = w(u, w.initial(3))
    y.square().mean().backward()
    res["grad_finite"] = all(bool(torch.isfinite(p.grad).all()) for p in w.parameters() if p.grad is not None)
    res["grad_params"] = sum(p.grad is not None for p in w.parameters())
    OUT["mamba3"] = res
    print(json.dumps(res, indent=1), flush=True)


def units():
    cfg = cfg_of()
    torch.manual_seed(1)
    RFT = importlib.import_module("artifacts.experiments.20261116_temporal_memory.record_ft")
    rec = M.Record(cfg).to(DEV)
    p = torch.randn(2, 30, 81, 16, device=DEV); first = torch.zeros(2, 30, dtype=torch.bool, device=DEV); first[:, 0] = True
    first[1, 11] = True
    a = rec.scan(p, first, rec.zero(2)); b = RFT.record(p, first, torch.zeros(2, 81, 2, device=DEV))
    res = {"record_equals_test2": max(rel(a[0], b[0]), rel(a[1], b[1]))}
    h = M.Harmoniser(["A", "B"], 1.0, 0.01).train()
    for v in (0.3, 0.2, 0.25):
        h({"A": torch.tensor(v), "B": torch.tensor(0.002)})
    mu = h.mu
    res["harmoniser_fixed_point"] = rel(h.log_sigma.exp(), (mu + (mu * mu + 4 * mu).sqrt()) / 2)
    res["harmoniser_no_parameters"] = len(list(h.parameters())) == 0
    model = M.M0(cfg).to(DEV)
    model.denoiser.checkpoint = True
    split = D.load_split(cfg)
    P = cfg["data"]["context_prefix"]
    streams = D.Streams(split["train"][:40], 4, P, seed=3, targets_offsets=D.frame_offsets(split["train"]))
    tmp = Path(tempfile.mkdtemp())
    ctl = C.Controller(model, cfg, tmp, DEV)
    chunk = streams.next(32)
    ctl.prepare(chunk, streams)
    rows = [1, 3]
    xx, q, local = TR.gaze_rows(ctl, rows, 32)
    _, _, ids = ctl.pending
    sid, tid = (ids // 32).cpu().numpy(), (ids % 32).cpu().numpy()
    want = [(int(np.searchsorted(rows, s)) * 32 + t) for s, t in zip(sid, tid) if s in rows]
    res["gaze_rows"] = sorted(local.cpu().tolist()) == sorted(want) and bool(torch.allclose(q.sum(-1), torch.ones_like(q[:, 0])))
    carry, cache = model.belief.initial(4, DEV), torch.zeros(4, P + 1, cfg["belief"]["width"], device=DEV)
    boot = TR.bootstrap_rows(cfg, 1, 0)
    with torch.random.fork_rng(devices=[DEV.index]):
        out = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg, boot[rows], 0,
                           None, rec.zero(2), TR.gaze_rows(ctl, rows, 32))
    terms = out[0]
    res["terms"] = {k: float(v) for k, v in terms.items()}
    res["encoder_norms"] = TR.encoder_grad_norms(model, terms)
    belief = tuple(model.belief.mixers.parameters())
    res["mamba_norms"] = {}
    for k in ("g1", "reward", "cont", "reward_forecast", "gaze_forecast"):
        grads = [g for g in torch.autograd.grad(terms[k], belief, retain_graph=True, allow_unused=True) if g is not None]
        res["mamba_norms"][k] = float(torch.stack([g.float().norm() for g in grads]).norm()) if grads else 0.0
    blocked = ("reward", "cont", "reward_forecast", "cont_forecast", "gaze_forecast")
    res["route_by_role_ok"] = all(res["encoder_norms"][k] == 0.0 for k in blocked) and \
        all(res["encoder_norms"][k] > 0 for k in ("g1", "tsp", "rec")) and all(res["mamba_norms"][k] > 0 for k in res["mamba_norms"])
    res["record_state_shape"] = list(out[3].shape)
    shutil.rmtree(tmp)
    OUT["units"] = res
    print(json.dumps(res, indent=1), flush=True)


def throughput(updates=30):
    """Run-scale cost: the trainer itself (16 streams, F5's micro-batching) for `updates` updates at 64 and at 256
    steps, route by role on and off; seconds per update over updates 2-25 and peak GiB, from the trainer's own log."""
    res = {}
    for route in (True, False):
        for steps in (64, 256):
            tmp = Path(tempfile.mkdtemp())
            torch.cuda.reset_peak_memory_stats()
            TR.p1(cfg_of(route), P7.WORLD_SEED, tmp, updates=updates, steps=steps, checkpoints=False)
            log = [json.loads(l) for l in (tmp / "log.jsonl").read_text().splitlines()]
            res[f"route_{route}_steps_{steps}"] = {"sec_per_update": log[-1]["sec_per_update"], "peak_gib": log[-1]["peak_gib"],
                                                   "g1": log[-1]["g1"], "sigma": log[-1]["sigma"]}
            print(json.dumps(res), flush=True)
            shutil.rmtree(tmp)
    OUT["throughput"] = res


if __name__ == "__main__":
    {"mamba3": mamba3, "units": units, "throughput": throughput}[sys.argv[1]]()
    f = HERE / "checks.json"
    prev = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps({**prev, **OUT}, indent=1) + "\n")
