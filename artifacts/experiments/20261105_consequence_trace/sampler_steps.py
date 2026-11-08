"""Stage B3 (PLAN.md): the sampler's step count on the frozen worlds. trace.py's base variant (same contexts, rows,
functionals, inputs and seeds: trace.contexts / functional / inputs / run); only edm.sampler.steps changes (3, 8, 16,
32; EDM's per-step churn gamma = min(S_churn / N, sqrt 2 - 1) follows N, as in model.sample). 3 steps must reproduce
the trace's production row by row (R0).

Usage: python -m artifacts.experiments.20261105_consequence_trace.sampler_steps OUT_DIR CKPT TRACE_DIR STONE_DIR"""
import copy, importlib, json, sys
from pathlib import Path

import numpy as np
import torch

T = importlib.import_module("artifacts.experiments.20261105_consequence_trace.trace")
STEPS = (3, 8, 16, 32)
KINDS = ("risky", "adjacent", "coal", "iron", "drink", "wood", "stone", "safe")


@torch.no_grad()
def main():
    out_dir, ckpt = Path(sys.argv[1]), T.ROOT / sys.argv[2]
    trace_dirs = {k: Path(sys.argv[4] if k == "stone" else sys.argv[3]) for k in KINDS}
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = T.M.M0(cfg).to(T.DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(T.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    enc = T.S.encode(model, L, cfg)
    hud, blocks = T.means(L, enc)
    ctx, _ = T.contexts(L, enc, np.random.default_rng(T.SEED))
    cfgs = {}
    for n in STEPS:
        cfgs[n] = copy.deepcopy(cfg)
        cfgs[n]["edm"]["sampler"]["steps"] = n
    res = {"checkpoint": str(ckpt.relative_to(T.ROOT)), "checkpoint_sha256": T.TR.file_sha256(ckpt),
           "script_sha256": T.TR.file_sha256(Path(__file__)), "trace_sha256": T.TR.file_sha256(Path(T.__file__)),
           "steps": STEPS, "first_sigma_hat": {n: T.schedule(cfgs[n])[0][0] * (1 + T.schedule(cfgs[n])[1][0]) for n in STEPS},
           "classes": {}}
    n_ctx = 128 // T.R
    for kind in KINDS:
        rows = ctx[kind]
        parts = ("hud", "cell") if kind in T.MINING else ("hud",)
        o, valid, row_ctx = {n: {} for n in STEPS}, [], []
        for b, start in enumerate(range(0, len(rows), n_ctx)):
            batch = rows[start:start + n_ctx]
            idx = torch.arange(len(batch)).repeat_interleave(T.R)
            row_ctx.append(start + idx)
            fw = {part: T.functional(kind, part, batch, enc, L, hud, blocks) for part in parts}
            fs = {}
            for part, (A, W, *_r) in fw.items():
                A_, W_ = A[idx].to(T.DEV), W[idx].to(T.DEV)
                fs[part] = lambda y, A_=A_, W_=W_: ((y.float() - A_) * W_).sum((-1, -2)).cpu()
            valid.append(torch.stack([fw[p_][5][idx] for p_ in parts], 1).all(1))
            inp = T.inputs(model, cfg, enc, batch, "base")
            for n in STEPS:                                              # same seed as the trace's base variant
                for k, v in T.run(model, cfgs[n], inp, fs, T.SEED + b, None, check=b == 0).items():
                    o[n].setdefault(k, []).append(v)
        flat = {f"n{n}_{k}": torch.cat(v).numpy() for n, d in o.items() for k, v in d.items()}
        valid, row_ctx = torch.cat(valid).numpy(), torch.cat(row_ctx).numpy()
        np.savez_compressed(out_dir / f"{kind}.npz", **flat, valid=valid, row_ctx=row_ctx)
        tz = np.load(trace_dirs[kind] / f"{kind}.npz")
        ep = np.array([rows[i][0] for i in row_ctx])
        est = lambda x: T.G.estimate(np.asarray(x, float), valid, ep)
        c = res["classes"][kind] = {"rows_valid": int(valid.sum()), "parts": {}}
        for part in parts:
            repro = float(np.abs(flat[f"n3_prod_{part}"] - tz[f"base_prod_{part}"]).max())
            first3 = est(flat[f"n3_first_{part}"])
            c["parts"][part] = {"R0_max_abs_diff_vs_trace_3_steps": repro, "first_3": first3, "by_steps": {
                n: {"first": est(flat[f"n{n}_first_{part}"]), "production": est(flat[f"n{n}_prod_{part}"]),
                    "first_gt_half": float(np.mean(flat[f"n{n}_first_{part}"][valid] > .5)),
                    "recovery_vs_first_3": est(flat[f"n{n}_prod_{part}"])["est"] / first3["est"] if first3["est"] else None}
                for n in STEPS}}
            print(kind, part, "repro", f"{repro:.2e}", {n: round(v["production"]["est"], 3) for n, v in c["parts"][part]["by_steps"].items()},
                  "first3", round(first3["est"], 3), flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
