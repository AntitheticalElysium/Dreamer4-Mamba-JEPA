"""Learned-fixation premise test A (PLAN.md): which of P1-F's remaining consequence deficits are the trained field's
(a training-signal deficit, the only kind a fixation chooser can reach), the sampler's, or hidden state's? GPU.
Instruments unchanged from the generation diagnosis (10-29): field_calibration (the field's conditional mean on context
classes defined from state t - 1 and a_{t-1}, never the outcome) and solver_sensitivity (production vs Heun 32 on the
same contexts and draws). Added classes: WOOD, SAPLING, STONE, COAL, IRON (awake, DO facing the block, holding the
tool it needs, counter below 9; functional: the counter's HUD cell, start -> start + 1). Mobs: S1's frozen readers
(own-gaze run) count each kind on RANDOM's real successors and on both samplers' samples.
Conditions: world 1 under random gaze; P1-F under random gaze and under the rule's gaze (its training placement).
Usage (project venv, GPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.decompose OUT_DIR [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
SS = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.solver_sensitivity")
LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
EXP = Path(__file__).resolve().parents[1]
OWN = EXP / "20261030_p1_guided/readings/own_gaze"
P1F = EXP / "20261030_p1_guided/run/ckpt_10.pt"
CONDITIONS = {"w1_random": (FC.CKPT, False, OWN / "s1_w1_random.reader.pt"),
              "p1f_random": (P1F, False, OWN / "s1_p1f_random.reader.pt"),
              "p1f_guided": (P1F, True, OWN / "s1_p1f_guided.reader.pt")}
COLLECT = {"wood": (5, 68, None), "sapling": (2, 67, None), "stone": (4, 69, 6), "coal": (8, 70, 6), "iron": (9, 71, 7)}
SS.ALGORITHMS = {"heun_32": SS.ALGORITHMS["heun_32"]}                     # 10-29 D's decisive arm only
PANEL = 1000                                                              # solver contexts per class (10-29 D: 1,000-1,500)


def collect_contexts(L, enc, rng):
    """COLLECT classes as FC.contexts rows: (episode, t, cell, start, start + 1, False), at most FC.N_MAX['drink'] each."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    e = np.concatenate([np.full(len(enc[i]["p"]) - FC.P, i) for i in range(len(enc))])
    t = np.concatenate([np.arange(FC.P, len(enc[i]["p"])) for i in range(len(enc))])
    g = s_off[e] + t - 1
    awake_do = (L["t_action"][a_off[e] + t - 1] == FC.DO) & ~L["s_sleep"][g].astype(bool)
    out = {}
    for c, (block, cell, tool) in COLLECT.items():
        v = L["s_inv"][g, LA.HUD_SRC[cell][1]].astype(int)
        m = awake_do & (face[g] == block) & (v < 9) & (True if tool is None else L["s_inv"][g, tool] > 0)
        rows = [(int(a), int(b), cell, int(s), int(s) + 1, False) for a, b, s in zip(e[m], t[m], v[m])]
        n = FC.N_MAX["drink"]
        out[c] = [rows[i] for i in rng.choice(len(rows), n, replace=False)] if len(rows) > n else rows
    return out


def value_means(L, enc, cells):
    """FC.value_means for any HUD counter: the mean real token of each cell per displayed value."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    p = torch.cat([enc[e]["p"][:, cells].float() for e in range(len(enc))])
    v = np.concatenate([np.stack([L[k][s_off[e]:s_off[e + 1], c] for k, c in (LA.HUD_SRC[x] for x in cells)], 1)
                        for e in range(len(enc))]).clip(min=0)
    return {cell: {int(k): p[v[:, j] == k, j].mean(0) for k in np.unique(v[:, j])} for j, cell in enumerate(cells)}


def recorded_gain(o, ctx, L):
    """FC.summary's recorded end-value rate reads health or drink only; this reads each row's own counter."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rows = [ctx[i] for i in o["row_ctx"]]
    after = np.array([L[LA.HUD_SRC[r[2]][0]][s_off[r[0]] + r[1], LA.HUD_SRC[r[2]][1]] for r in rows])
    ep = np.array([r[0] for r in rows])
    return G.estimate((after == np.array([r[4] for r in rows])).astype(float), np.ones(len(ep), bool), ep)


@torch.no_grad()
def kinds(reader, p, v):
    """S1.3's count readout: kind per map cell (S1's KEEP cells), counted per frame."""
    k = torch.cat([reader(torch.cat([p[i][:, :63].reshape(-1, p.shape[-1]).half().to(FC.DEV),
                                     v[i].repeat_interleave(63, 0).half().to(FC.DEV)], 1))["kind"].argmax(1)
                   .reshape(len(i), 63).cpu() for i in torch.arange(len(p)).split(2048)]).numpy()[:, S.KEEP[:63]]
    return {kn: (k == j).sum(1) for j, kn in enumerate(G.KINDS, 1)}


def main():
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 20 if smoke else 0)
    res = {"sigmas": FC.SIGMAS, "rows_per_context": FC.R, "solver_panel": PANEL, "algorithms": list(SS.ALGORITHMS)}
    for name, (path, guided, reader_path) in CONDITIONS.items():
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(FC.DEV); model.load_state_dict(ck["model"]); model.eval()
        del ck
        enc = S.encode(model, L, cfg, windows=PM.window_of_state(L, rule) if guided else None)
        means = {**FC.value_means(L, enc), **value_means(L, enc, [c for _, c, _ in COLLECT.values()])}
        ctx = {**FC.contexts(L, enc, np.random.default_rng(2029)), **collect_contexts(L, enc, np.random.default_rng(2031))}
        ctx = {c: [r for r in rows if r[3] in means[r[2]] and r[4] in means[r[2]]][:64 if smoke else None]
               for c, rows in ctx.items()}
        reader = S.load_reader(reader_path, path, FC.DEV)
        res[name] = {"checkpoint": str(path), "sha256": TR.file_sha256(path), "gaze": "rule" if guided else "random"}
        for c, rows in ctx.items():
            if not rows:
                res[name][c] = {"contexts": 0}
                continue
            gen = torch.Generator(device=FC.DEV).manual_seed(2031)
            o = FC.run(model, enc, rows, means, cfg, gen)
            r = FC.summary(o, rows, L, enc)
            if c in COLLECT:
                r["recorded_reached_end_value"] = recorded_gain(o, rows, L)
            panel = rows[:PANEL]
            so = SS.run(model, enc, panel, means, cfg, gen, keep=c == "random")
            ep = np.array([panel[i][0] for i in so["row_ctx"]])
            est = lambda x: G.estimate(np.asarray(x, float), np.ones(len(ep), bool), ep)
            r["solver_panel"] = {"contexts": len(panel), **{a: est(so[f"f_{a}"]) for a in ["production", *SS.ALGORITHMS]}}
            if c == "random":
                real = [torch.stack([enc[panel[i][0]][k][panel[i][1]] for i in so["row_ctx"]]) for k in ("p", "v")]
                src = {"real": real, **{a: [torch.as_tensor(so[f"{k}_{a}"]) for k in ("p", "v")]
                                        for a in ["production", *SS.ALGORITHMS]}}
                r["mob_counts"] = {s: {k: est(x) for k, x in kinds(reader, *pv).items()} for s, pv in src.items()}
            res[name][c] = r
            print(name, c, len(rows), "f rec", round(r["f_recorded"]["est"], 3), "field 6.67",
                  round(r["f_recorded"]["est"] + r["by_sigma"]["6.67"]["bias"]["est"], 3), "prod",
                  round(r["f_production"]["est"], 3), "heun32", round(r["solver_panel"]["heun_32"]["est"], 3), flush=True)
            (out_dir / "results.json").write_text(json.dumps(res, indent=1))
        del model, enc
        torch.cuda.empty_cache()
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
