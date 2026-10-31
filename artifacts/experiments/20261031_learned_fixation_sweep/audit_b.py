"""Audits of progress.py before B's launch (user 10-31, after Codex's partial review: the third geometry was not LESS's
Gamma; now replaced). CPU, fp32, Mamba's reference recurrence; progress.py's own functions throughout.
1. Finite differences: for each geometry's direction d (euclid g, adam_linear pre * g, adam_step Gamma(g)) of a real
   condition gradient g (drink contexts, counter window), the measured decrease of a real gaze-free target (drink
   counter cell, fresh dev frames, the same noise draws) after theta -> theta - eta d, against the predicted eta <d, g_T>,
   at two step sizes (sign, flattening order, linearity); the per-module dots sum to the total. The fixed target
   (stop-gradient tokens, as training) is checked with the encoder left out of the step, since a step on the encoder
   moves the tokens it holds fixed (first run of this audit: measured 3-5x the prediction on the encoder's part alone);
   the full target (tokens move) with every parameter stepped.
2. Determinism: the same condition twice under the same seed gives the same gradient (common random numbers).
3. adam_step: zero at g = 0 (the momentum-alone update is subtracted exactly); per-parameter step bounded where the
   second moment is ~0 (saturation, the reason for the geometry).
4. drift(): residual ~0 when the new tokens are an exact affine map of the old, ~sigma^2 with added noise.
5. P1-F's history gaze on train-400: the share of frames placed by the rule (training: 13.1%).
Usage (project venv, CPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.audit_b OUT_JSON
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

PG = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.progress")
P, H = PG.P, PG.H


def main():
    out, res = Path(sys.argv[1]), {}
    assert PG.DEV.type == "cpu", "run with CUDA hidden"
    S.DEV = PG.EV.DEV = PG.DEV
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cfg0 = TR.load_cfg()
    split = D.load_split(cfg0)
    targets = np.load(TR.ROOT / cfg0["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    Lt = dict(np.load(S.LABELS / "s0_train400_labels.npz"))
    where = {(e.shard, e.slot): i for i, e in enumerate(split["train"])}
    keep = np.array([(int(s), int(o)) in where for s, o in zip(Lt["ep_shard"], Lt["ep_slot"])])
    Ld = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 40)
    rule = {int(k): tuple(v) for k, v in json.loads((PG.PM.HERE / "map.json").read_text())["rule"].items()}
    win = PG.PM.window_of_state(Lt, rule)
    s_off = np.r_[0, np.cumsum(Lt["ep_T"] + 1)]
    later = np.concatenate([np.arange(s_off[e] + 1, s_off[e + 1]) for e in np.flatnonzero(keep)])
    res["5_p1f_rule_share_train400"] = float((win[later - 1, 0] >= 0).mean())

    rc = PG.classes_at(Lt, "drink", P, H - 1, keep)[:4]
    rt = PG.classes_at(Ld, "drink", 2 * P - 1, 0, np.ones(len(Ld["ep_T"]), bool))[:4]
    fx_t, fx_d = PG.gaze(Lt, cfg0), PG.gaze(Ld, cfg0)
    offsets = D.frame_offsets(split["train"])
    td = {e: PG.load(split["train"], where[(int(Lt["ep_shard"][e]), int(Lt["ep_slot"][e]))], fx_t[e], max(t for e_, t in rc if e_ == e) + H - 1, offsets)
          for e in {e for e, _ in rc}}
    dd = {e: PG.load(S.episodes(Ld), e, fx_d[e], max(t for e_, t in rt if e_ == e)) for e in {e for e, _ in rt}}
    tc = {e: {t for e_, t in rc if e_ == e} for e in td}
    dc = {e: {t - P + 1 for e_, t in rt if e_ == e} for e in dd}
    d_ = PG.FC.DIRS[Ld["s_dir"]]
    so = np.r_[0, np.cumsum(Ld["ep_T"] + 1)]
    cells = torch.as_tensor([(65, (3 + d_[so[e] + t - 1, 0]) * 9 + 4 + d_[so[e] + t - 1, 1]) for e, t in rt])
    for k in (2, 10):
        ck = torch.load(TR.ROOT / f"artifacts/saccade_m0/p1_world1/ckpt_{k:02d}.pt", weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg); model.load_state_dict(ck["model"]); model.train()
        for mx in model.belief.mixers:
            mx.backend = "reference"
        params, slices, pre, adam_step = PG.geometry(model, ck, cfg)
        pd, pt = PG.prepass(model, cfg, dd, dc), PG.prepass(model, cfg, td, tc)
        tchunk, tcarry, tcache = PG.batch(dd, pd, rt, -2 * P + 1, 1, -P + 1)
        tgt = {f: (lambda f=f: PG.target_terms(model, cfg, tchunk, tcarry, tcache, cells, torch.Generator().manual_seed(7), full=f)["counter"])
               for f in (False, True)}
        L0 = {f: tgt[f]() for f in tgt}
        gT = {f: PG.flat(torch.autograd.grad(L0[f], params, allow_unused=True), params) for f in tgt}
        chunk, carry, cache = PG.batch(td, pt, rc, -P, H, 0)
        chunk["fixation"][:, P] = (7, 2)

        def cond():
            torch.manual_seed(11)
            terms, _, _ = TR.p1_losses(model, chunk, carry, cache, targets, cfg)
            loss = model.harmoniser(TR.groups(terms)) + terms.get("noise", 0)
            return PG.flat(torch.autograd.grad(loss, params, allow_unused=True), params)
        g, g2 = cond(), cond()
        r = {"2_determinism_max_abs_diff": float((g - g2).abs().max()), "grad_norm": float(g.norm()),
             "3_adam_step_at_zero_max_abs": float(adam_step(torch.zeros_like(g)).abs().max()),
             "3_adam_step_max_over_lr": float((adam_step(g) / pre.new_tensor(cfg["optimizer"]["lr"])).abs().max()),
             "pre_range": [float(pre.min()), float(pre.median()), float(pre.max())], "1_finite_differences": {}}
        theta = torch.nn.utils.parameters_to_vector(params).detach().clone()
        a0, b0 = slices[0]                                               # perception
        for name, d0 in (("euclid", g), ("adam_linear", pre * g), ("adam_step", adam_step(g))):
            for full in (False, True):
                d = d0.clone()
                if not full:
                    d[a0:b0] = 0
                pred = float(d @ gT[full])
                mods = sum(float(d[a:b] @ gT[full][a:b]) for a, b in slices)
                fd = []
                for scale in (1.0, 0.25):
                    eta = scale * 1e-3 / float(d.abs().max())            # largest parameter change 1e-3 (x scale)
                    with torch.no_grad():
                        torch.nn.utils.vector_to_parameters(theta - eta * d, params)
                        Lp = tgt[full]()
                        torch.nn.utils.vector_to_parameters(theta, params)
                    fd.append({"eta": eta, "predicted_decrease": eta * pred, "measured_decrease": float(L0[full] - Lp),
                               "ratio": float(L0[full] - Lp) / (eta * pred) if pred else None})
                r["1_finite_differences"][f"{name}_{'full' if full else 'fixed_no_encoder'}"] = {
                    "dot": pred, "module_sum_minus_total": mods - pred, "steps": fd,
                    "encoder_part_fixed_vs_full": [float(d0[a0:b0] @ gT[False][a0:b0]), float(d0[a0:b0] @ gT[True][a0:b0])]}
        res[f"w1_ckpt{k:02d}"] = r
        print(k, json.dumps(r, indent=1), flush=True)

    rng = np.random.default_rng(0)
    n = 400
    prev = {"p": torch.randn(n, 81, 16), "g": torch.randn(n, 9, 64), "v": torch.randn(n, 8)}
    A = {k: (torch.randn(x.shape[-1], x.shape[-1]) / x.shape[-1] ** 0.5, torch.randn(x.shape[-1])) for k, x in prev.items()}
    exact = {k: x @ A[k][0] + A[k][1] for k, x in prev.items()}
    noisy = {k: x + 0.1 * torch.randn_like(x) for k, x in exact.items()}
    odd = np.arange(n) % 2 == 1
    cls = rng.integers(0, 10, (n, 81))
    res["4_drift_exact_affine"] = PG.drift(prev, exact, odd, cls)[0]
    res["4_drift_affine_plus_noise_0.01"] = PG.drift(prev, noisy, odd, cls)[0]
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("4", "5"))}, indent=1))


if __name__ == "__main__":
    main()
