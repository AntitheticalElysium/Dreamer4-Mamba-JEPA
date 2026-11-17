"""P1.5's evaluation queue (PLAN, readings): F6.1's queue (../20261114_p1f61/evaluation.py: F5's readout tools, then the
clock, trajectory, follow-up, sleep-split, entity and comparison stages) on the selected checkpoint, F6.1 the
comparison arm, every tool under the record guard. Before any reading: the adapter check (Codex's review) on this
checkpoint, then the smoke pass of F5's tools; a failure stops the queue (training and selection stand).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_p15.evaluation CHECKPOINT       (the worker's call)
    ... evaluation --tool MODULE ARGS...           one tool under the guard
    ... evaluation --stage NAME CHECKPOINT READINGS
    ... evaluation --check CHECKPOINT              the adapter check alone
"""
import importlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
F61 = HERE.parent / "20261114_p1f61"
E61 = importlib.import_module("artifacts.experiments.20261114_p1f61.evaluation")   # imports F5's readout queue first
R = E61.R
ME = f"{__package__}.evaluation"


def guard():
    """Codex's review: the sampler silently drops the record when a caller passes only Mamba's memory tokens. Under the
    guard a denoiser with a record read refuses them; Mamba's side already raises (percept_embed needs the record)."""
    embed = M.Denoiser.embed

    def guarded(self, x, memory, fix):
        if self.record_read is not None and not isinstance(memory, tuple):
            raise RuntimeError("the spatial history is missing at a denoiser call (P1.5)")
        return embed(self, x, memory, fix)
    M.Denoiser.embed = guarded


def steps(checkpoint, readings, smoke):
    """F6.1's steps, each tool under the guard, F6.1's stages through this run's dispatcher."""
    rows = []
    for name, command, out in E61.steps(checkpoint, readings, smoke):
        assert command[0] == "-m"
        command = ["-m", ME, "--stage", *command[2:]] if command[1] == "artifacts.experiments.20261114_p1f61.stages" \
            else ["-m", ME, "--tool", *command[1:]]
        rows.append((name, [c.replace("f61_4steps=", "p15_4steps=") for c in command], out))
    return rows


def stage(name, checkpoint, readings):
    """F6.1's stages (stages.py) pointed at this run; entities and the comparison labelled p15 beside F6.1."""
    k = int(checkpoint.stem[-2:])
    assert checkpoint == (HERE / f"run/ckpt_{k:02d}.pt").resolve()
    readings.mkdir(parents=True, exist_ok=True)
    if name == "clocks":
        RD = importlib.import_module("artifacts.experiments.20261114_p1f61.readout")
        RD.HERE, RD.OUT = HERE, readings
        for j in sorted({2, 3, k}):
            RD.main(j, RD.STAGES)
    elif name == "entities":
        EN = importlib.import_module("artifacts.experiments.20261111_p1f5.entities")
        EN.OUT = readings / "dissection"
        EN.OUT.mkdir(parents=True, exist_ok=True)
        EN.DS.CKPTS = dict(EN.DS.CKPTS, f5=checkpoint)
        EN.main()
        path = EN.OUT / "entities.json"
        res = json.loads(path.read_text())
        res["p15"] = res.pop("f5")
        path.write_text(json.dumps(res, indent=1) + "\n")
    elif name == "compare":
        CP = importlib.import_module("artifacts.experiments.20261111_p1f5.compare")
        CP.HERE = HERE
        for arm, root in (("f61", F61 / "readings"), ("p15", readings)):
            CP.CURVES[arm] = root / "consequences/curves/raw.npz"
            CP.INTERFACE[arm] = root / "interface.json"
            CP.RECURSION[arm] = root / "recursion_4/raw.npz"
        CP.main()
    else:                                                                # trajectory, movement, mining, sleep, sleep_split
        ST = importlib.import_module("artifacts.experiments.20261114_p1f61.stages")
        ST.HERE = HERE
        ST.main(name, checkpoint, readings)


@torch.no_grad()
def check(model, cfg, enc, L, dev):
    """The adapter on real call paths, against the record scanned contiguously from each episode's start (from encode's
    stored tokens): history in chunks, encode's stored states, before / window, embed against percept_embed with the
    raw record, and interface.futures (the path most tools share) captured at its denoiser and outcome-head calls:
    every context pass and draw must read the record through the previous frame, the outcome heads the record after
    the frame scored (generated or recorded). Codex's dependence test: a changed frame t moves the read after it and
    leaves the denoiser's record for frame t identical. Last, the guard refuses memory tokens alone."""
    IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    res, P = {}, IF.P
    zero = lambda n: torch.zeros(1, n, dtype=torch.bool, device=dev)
    full = {}
    for e, rec in enumerate(enc):
        p = rec["p"][None].to(dev).float()
        full[e] = model.record.scan(p, zero(p.shape[1]), model.record.zero(1, dev))   # (through previous, after) [1, T, ..]
    e, T = 0, len(enc[0]["p"])
    rin, rst = full[e]
    p = enc[e]["p"][None].to(dev).float()
    reads, kept, last = S.history(model, p, keep=[0, 1, T - 1], chunk=7)
    rel = lambda a, b: float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp(min=1e-30))
    res["history_chunked_reads"] = rel(reads, model.belief.record_in(rst))
    res["history_chunked_kept"] = rel(kept, rin[:, [0, 1, T - 1]])
    res["history_chunked_state"] = rel(last, rst[:, -1])
    res["encode_states_vs_stored_tokens"] = max(rel(r["record_every"], full[i][0][0, ::S.EVERY]) for i, r in enumerate(enc))
    js = sorted({0, 1, S.EVERY - 1, S.EVERY, S.EVERY + 1, T - 1})
    res["before"] = rel(S.before(model, enc, [(e, j) for j in js]), rin[0, js])
    w = S.window(model, enc, [(e, P), (e, T - 1), (e, 3)], P)
    res["window"] = max(rel(w[0], rin[0, 0:P + 1]), rel(w[1], rin[0, T - 1 - P:T]), rel(w[2, P - 3:], rin[0, 0:4]))
    res["window_zero_before_episode"] = float(w[2, :P - 3].abs().max())
    with TR.autocast(dev, cfg):
        Pq = M.Percept(*(enc[e][k][None, :T].to(dev).float() for k in ("p", "g", "v")), enc[e]["fix"][None, :T].to(dev))
        res["embed_vs_percept_embed"] = rel(S.embed(model, Pq, model.belief.record_in(rst)), model.belief.percept_embed(Pq, rst))
    q = p.clone()
    q[:, T // 2] += 0.5
    rq, sq = model.record.scan(q, zero(T), model.record.zero(1, dev))
    res["dependence_denoiser_record_at_t_unchanged"] = float((rq[:, T // 2] - rin[:, T // 2]).abs().max())
    res["dependence_read_after_t_changed"] = float((model.belief.record_in(sq[:, T // 2]) - model.belief.record_in(rst[:, T // 2])).abs().max())
    # interface.futures, captured
    got = {"cond": [], "draw": [], "head": []}
    ctx, draw, emb = M.context_pass, M.sample, S.embed
    M.context_pass = lambda den, x, a, memory, *r, **k: (got["cond"].append(memory[1].clone()), ctx(den, x, a, memory, *r, **k))[1]
    M.sample = lambda den, shapes, a, memory, *r, **k: (lambda out: (got["draw"].append((memory[1].clone(), out[0].clone())), out)[1])(
        draw(den, shapes, a, memory, *r, **k))
    S.embed = lambda model_, P_, reads=None, alone=False: (got["head"].append((P_.p.clone(), reads.clone())), emb(model_, P_, reads, alone))[1]
    try:
        tuples = [(e, P), (e, T - 1), (1, max(P, len(enc[1]["p"]) // 2))]
        IF.futures(model, enc, L, tuples, 2, cfg, torch.Generator(device=dev).manual_seed(0))
    finally:
        M.context_pass, M.sample, S.embed = ctx, draw, emb
    rows = [i for i in range(len(tuples)) for _ in range(2)]
    expect = lambda dt: torch.stack([full[tuples[i][0]][0][0, tuples[i][1] + dt] for i in rows])
    res["futures_context_record"] = max(rel(got["cond"][P + dt][:, 0], expect(dt)) for dt in range(-P, 0))
    (mem_rec, out_p), = got["draw"]
    res["futures_draw_record"] = rel(mem_rec[:, 0], expect(0))
    (pg, reads_g), (pr, reads_r) = got["head"]

    def after(pp):                                                       # the read after frame pp from the draw's record
        with TR.autocast(dev, cfg):
            return model.belief.record_in(model.record.scan(pp[:, None].float(), torch.zeros(len(pp), 1, dtype=torch.bool, device=dev),
                                                            mem_rec[:, 0])[1][:, 0])
    res["futures_generated_frame_is_drawn"] = rel(pg, out_p[:, 0])
    res["futures_head_generated_read"] = rel(reads_g, after(pg))
    res["futures_head_recorded_read"] = rel(reads_r, after(pr))
    res["futures_recorded_frame_is_t"] = rel(pr, torch.stack([enc[tuples[i][0]]["p"][tuples[i][1]] for i in rows]).to(dev))
    guard()
    x = tuple(enc[e][k][None, :1].to(dev).float() for k in ("p", "g", "v"))
    m = model.belief.memory(enc[e]["bp"][None, :1].to(dev).float())
    try:
        model.denoiser.embed(x, m, enc[e]["fix"][None, :1].to(dev))
        res["guard_refuses_tokens_alone"] = False
    except RuntimeError:
        res["guard_refuses_tokens_alone"] = True
    exact = ("history_chunked_reads", "history_chunked_kept", "history_chunked_state", "embed_vs_percept_embed",
             "futures_generated_frame_is_drawn", "futures_head_generated_read", "futures_head_recorded_read", "futures_recorded_frame_is_t")
    stored = ("encode_states_vs_stored_tokens", "before", "window", "futures_context_record", "futures_draw_record")
    res["tolerances"] = {"exact": [list(exact), 1e-5], "against stored fp16 tokens (a wrong frame is O(1))": [list(stored), 1e-2]}
    res["passed"] = bool(all(res[k] < 1e-5 for k in exact) and all(res[k] < 1e-2 for k in stored)
                         and res["window_zero_before_episode"] == 0 and res["dependence_denoiser_record_at_t_unchanged"] == 0
                         and res["dependence_read_after_t_changed"] > 0 and res["guard_refuses_tokens_alone"])
    return res


def run_check(checkpoint):
    IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
    model, cfg = IF.load(str(checkpoint))
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    eps = [i for i, t in enumerate(L["ep_T"]) if 150 <= t <= 400][:2]   # past two of encode's stored states, short scans
    L = importlib.import_module("artifacts.experiments.20261107_danger_adversarial.sleep").subset(L, eps)
    res = check(model, cfg, S.encode(model, L, cfg), L, S.DEV)
    res["episodes"] = eps
    res.update(checkpoint=str(checkpoint), checkpoint_sha256=TR.file_sha256(checkpoint))
    (HERE / "eval_checks.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1), flush=True)
    if not res["passed"]:
        raise RuntimeError("record adapter check failed (eval_checks.json)")


def queue(checkpoint, out, smoke):
    R.steps, R.HERE = steps, HERE
    sys.argv = [sys.argv[0], str(checkpoint), "--out", str(out)] + (["--smoke"] if smoke else [])
    R.main()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    if sys.argv[1] == "--tool":
        guard()
        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    elif sys.argv[1] == "--stage":
        guard()
        stage(sys.argv[2], Path(sys.argv[3]).resolve(), Path(sys.argv[4]).resolve())
    elif sys.argv[1] == "--check":
        run_check(Path(sys.argv[2]).resolve())
    else:
        ck = Path(sys.argv[1]).resolve()
        run_check(ck)
        queue(ck, HERE / "checks/smoke", True)
        queue(ck, HERE / "readings", False)
