"""Follow-ups 2 and 3 (PLAN.md) on P1-F4's checkpoint, with its own sampler (K = 4 shortcut evaluations), its own readers
(curves.fit_readers' protocol) and S1 reader. Exp 3's bank, pair_state, conditions, factorial and shams
(../20261106_determinant_routes/routes.py) and T1's movement pairs and projection (../20261106_decision_levels/frozen.py),
unchanged; exp 3's EDM-specific probes dropped.
Usage (project venv): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.routes_f4 routes|movement
  python -m artifacts.experiments.20261107_f4_followups.routes_f4 analyse   (CPU: readings/routes_f4.pt -> routes_f4.json)"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
FZ = importlib.import_module("artifacts.experiments.20261106_decision_levels.frozen")
CV = RT.CV
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings"
CKPT = "artifacts/experiments/20261106_p1f4/run/ckpt_10.pt"
K_MOVE = 16


def setup():
    model, cfg = CV.IF.load(CKPT)
    assert model.denoiser.shortcut and model.denoiser.dense and cfg["edm"]["sampler"]["steps"] == 4
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    return model, cfg, L, S.encode(model, L, cfg)


def sample(model, cfg, shapes, action, bp, fx, cache, n, seed):
    """n futures of one member's inputs (exp 3's call: common random numbers through the seed)."""
    a = torch.full((n, 1), action, device=CV.DEV, dtype=torch.long)
    mem = model.belief.memory(bp)[:, None].expand(n, -1, -1, -1)
    with TR.autocast(CV.DEV, cfg):
        return M.sample(model.denoiser, shapes, a, mem, fx.expand(n, -1)[:, None], RT.stack(cache, n), cfg,
                        torch.Generator(device=CV.DEV).manual_seed(seed))


@torch.no_grad()
def routes():
    model, cfg, L, enc = setup()
    with torch.enable_grad():
        R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(CKPT).parent / "s1/results.reader.pt", TR.ROOT / CKPT, CV.DEV)
    shapes, lo, T = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")], cfg["edm"]["inference_context_sigma"], CV.transitions(L)
    res = []
    for i, r in enumerate(torch.load(RT.HERE / "readings/bank.pt", weights_only=False)):
        d = CV.IF.DIRS[L["s_dir"][int(T["g"][r["j"]])]]
        r["face_cell"] = int((3 + d[0]) * 9 + 4 + d[1])
        _, cons = RT.sites(r["kind"], r)
        P2, Ps, b, bp, caches, fx = RT.pair_state(model, cfg, enc, r)
        sham = {"tokens_max_abs": float((P2.p[0].float() - enc[r["e"]]["p"][r["t"] - 1].to(CV.DEV).float()).abs().max()),
                "belief_rel": float((bp[2] - enc[r["e"]]["bp"][r["t"]].to(CV.DEV).float()).norm() / bp[2].norm()),
                "cache_rel": max(float((a - c).float().norm() / c.float().norm()) for (a, _), (c, _) in
                                 zip(caches[lo][0]["kv"], caches["sham"][0]["kv"]))}
        assert sham["tokens_max_abs"] < 0.04, ("natural member's tokens differ from S1 encode", sham)
        gen = []
        for ck, mm in RT.COND:
            xs = sample(model, cfg, shapes, r["actions"][ck], bp[mm:mm + 1], fx, caches[lo][ck], RT.K, r["j"])
            if (ck, mm) == (0, 0) and i % 25 == 0:
                again = sample(model, cfg, shapes, r["actions"][ck], bp[mm:mm + 1], fx, caches[lo][ck], RT.K, r["j"])
                assert all(torch.equal(u, v) for u, v in zip(xs, again)), "generation sham differs"
                sham["generation_repeat"] = "bitwise equal"
            rd = CV.read(R, s1, xs[0][:, 0].float().cpu(), xs[2][:, 0].float().cpu(), np.full(RT.K, r["face_cell"]))
            gen.append({"asleep": rd["asleep"], "slot": rd[f"hud{cons[0]}"] if r["kind"] in RT.SLOT else None, "energy": rd["hud66"]})
        rec = CV.read(R, s1, Ps.p.float().cpu(), Ps.v.float().cpu(), np.full(2, r["face_cell"]))
        res.append({"kind": r["kind"], "e": r["e"], "t": r["t"], "j": r["j"], "stratum": r["stratum"], "truth": r["truth"], "sham": sham,
                    "rec": {"asleep": rec["asleep"], "slot": rec[f"hud{cons[0]}"] if r["kind"] in RT.SLOT else None}, "gen": gen})
        if len(res) % 50 == 0:
            print("pairs", len(res), r["kind"], {k: (round(v, 5) if isinstance(v, float) else v) for k, v in sham.items()}, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"pairs": res, "readers": valid}, OUT / "routes_f4.pt")
    analyse()


@torch.no_grad()
def movement():
    model, cfg, L, enc = setup()
    shapes, lo = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")], cfg["edm"]["inference_context_sigma"]
    rows = []
    for r in torch.load(FZ.OUT / "movement_bank.pt", weights_only=False):
        P2, Ps, b, bp, caches, fx = RT.pair_state(model, cfg, enc, r)
        A, B = (FZ.A_.vector(Ps.p[m].float(), Ps.v[m].float(), "sleep") for m in (0, 1))
        f = []
        for m in (0, 1):
            xs = sample(model, cfg, shapes, r["actions"][m], bp[m:m + 1], fx, caches[lo][m], K_MOVE, r["j"])
            f.append((((FZ.A_.vector(xs[0][:, 0].float(), xs[2][:, 0].float(), "sleep") - A) * (B - A)).sum(-1) / (B - A).square().sum()).cpu().numpy())
        rows.append({"e": r["e"], "j": r["j"], "d": float((B - A).norm()), "f": np.stack(f)})
    moved = np.array([(x["f"] < 0.5).mean(1) for x in rows])            # [pairs, member]
    out = {"pairs": len(rows), "episodes": len({x["e"] for x in rows}), "K": K_MOVE,
           "moved (member 0 = move, member 1 = NOOP)": moved.mean(0).round(3).tolist(),
           "pairs with every move future moved / every NOOP future still": [int((moved[:, 0] == 1).sum()), int((moved[:, 1] == 0).sum())],
           "f_mean (0 = move successor, 1 = NOOP successor)": np.array([x["f"].mean(1) for x in rows]).mean(0).round(3).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "movement_f4.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


def analyse():
    """routes.analyse's factorial part on F4's pairs (no availability probes), beside F3's classes."""
    res = torch.load(OUT / "routes_f4.pt", weights_only=False)["pairs"]
    f3 = json.loads((RT.HERE / "readings/routes.json").read_text())
    mx = {k: max(r["sham"][k] for r in res) for k in ("tokens_max_abs", "belief_rel", "cache_rel")}
    readings = {"shams": {**mx, "generation_repeats_bitwise": sum("generation_repeat" in r["sham"] for r in res)}}
    for kind in RT.KINDS:
        P_ = [r for r in res if r["kind"] == kind]
        ep = np.array([r["e"] for r in P_])
        if kind in RT.SLOT:
            n = np.array([r["truth"]["count"][1] for r in P_])
            ev = lambda r, c, i: (r["gen"][c]["slot"] == n[i] + 1).mean()
            tr = np.full(len(P_), -1.0)
        else:
            ev = lambda r, c, i: (r["gen"][c]["asleep"] < 0.5).mean()
            tr = np.full(len(P_), 1.0)
        E = np.array([[ev(r, c, i) for c in range(4)] for i, r in enumerate(P_)])
        rd = {"pairs": len(P_), "episodes": int(len(np.unique(ep))), "factorial": RT.factorial(E, tr, ep),
              "f3_classes": f3[kind]["factorial"]["classes"], "f3_cells": {c: v["est"] for c, v in f3[kind]["factorial"]["cells"].items()}}
        for s in sorted({r["stratum"] for r in P_ if r["stratum"] is not None}):
            m = np.array([r["stratum"] == s for r in P_])
            if len(np.unique(ep[m])) >= 5:
                rd[f"factorial_{s}"] = RT.factorial(E[m], tr[m], ep[m])
        rd["recorded_reader_check"] = float(np.mean([(r["rec"]["asleep"] < 0.5).tolist() == [False, True] if kind not in RT.SLOT
                                                     else (r["rec"]["slot"] == [n[i] + 1, n[i]]).all() for i, r in enumerate(P_)]))
        readings[kind] = rd
        print(kind, {c: round(v["est"], 3) for c, v in rd["factorial"]["cells"].items()}, json.dumps(rd["factorial"]["classes"]),
              "| F3", rd["f3_cells"], flush=True)
    (OUT / "routes_f4.json").write_text(json.dumps(readings, indent=1, default=str))


if __name__ == "__main__":
    if sys.argv[1] != "analyse":
        assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    {"routes": routes, "movement": movement, "analyse": analyse}[sys.argv[1]]()
