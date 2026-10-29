"""PLAN v2 test F (consequence catalogue), with test A's instrument and test D's samplers. Descriptive. One GPU pass.

Question: is the field's underweighting (A: damage, drink) specific to some consequences, or general? Events differ
in frequency, footprint, determinism and observability at once (REVIEW sec. 6), so the catalogue describes; it does not
isolate a cause.
Classes, each a function of the state at t-1 and a_{t-1} (in the conditioning), rules from game_logic:
  MOVE     a_{t-1} in LEFT/RIGHT/UP/DOWN, awake, target cell in view not solid, not lava, no mob: the view shifts by
           one cell (move_player). Common, deterministic, footprint ~60 cells.
  BLOCKED  the same actions toward a solid or out-of-bounds cell: no shift (direction only). Common, deterministic.
  WOOD     awake, facing a tree, DO, wood <= 8: wood + 1 (do_action). Deterministic, HUD cell 68.
  SAPLING  awake, facing grass with no mob on it, DO, sapling <= 8: sapling + 1 with probability 0.1 (do_action's
           uniform < 0.1). Stochastic with a known rate, HUD cell 67.
  TABLE    awake, PLACE_TABLE, wood >= 2, faced cell not solid: a crafting table on the faced cell (place_block).
           Deterministic, one map cell.
  DRINK    as in A (replicates A on fresh rows).
Functional (fixed per context before any output, linear, so A's law-of-total-expectation identity holds):
f(y) = sum_c (y_c - a_c) . (b_c - a_c) / sum_c |b_c - a_c|^2 over the consequence's cells, a = "no consequence",
b = "consequence": HUD cells: mean real token at the starting / consequence count; MOVE / BLOCKED: a = the context
frame t-1's tokens, b = those tokens shifted by the move (cells whose source is off-screen, the player's cell and the
cell it left excluded); TABLE: a = the faced cell's token at t-1, b = the mean real table token at that screen cell.
Readouts as A (bias by sigma, recorded and production f) plus Heun 32 (D). Instrument checks: |bias| < 0.01 at sigma
0.05; recorded outcomes as defined (deterministic classes >= 99%; SAPLING ~0.10).
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.catalogue OUT_DIR [N_EPISODES]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
SS = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.solver_sensitivity")
ROOT = Path(__file__).resolve().parents[3]
DEV, P, R = FC.DEV, FC.P, 2
SIGMAS = FC.SIGMAS
SOLID = (3, 4, 5, 8, 9, 10, 11, 12, 15, 16)                              # constants.SOLID_BLOCKS
GRASS, TREE, WATER, LAVA, OOB, TABLE = 2, 5, 3, 14, 1, 11
DO, PLACE_TABLE = 5, 8
HUD = {"wood": (68, 1, 0), "sapling": (67, 1, 5), "drink": (65, 0, 2)}   # cell, labels array (0 s_body, 1 s_inv), column
N_MAX = 2000
MIN_SEPARATION = 1.0                                                     # |b - a|^2 below this: f undefined, row dropped


def contexts(L, enc, rng):
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    pos, view, sleep = L["s_pos"].astype(int), L["s_view"], L["s_sleep"]
    d = FC.DIRS[L["s_dir"]]
    face = view[np.arange(len(view)), 3 + d[:, 0], 4 + d[:, 1]]
    mob_at = lambda g, cell: any((L[f"s_{k}_m"][g] & (L[f"s_{k}_p"][g].astype(int) == cell).all(-1)).any()
                                 for k in ("zombies", "cows", "skeletons", "arrows"))
    out = {c: [] for c in ("move", "blocked", "wood", "sapling", "table", "drink")}
    frames = 0
    for e in range(len(enc)):
        for t in range(P, len(enc[e]["p"])):
            frames += 1
            g, a = s_off[e] + t - 1, int(L["t_action"][a_off[e] + t - 1])
            if sleep[g]:
                continue
            if 1 <= a <= 4:
                dd = FC.DIRS[a]
                tgt = view[g, 3 + dd[0], 4 + dd[1]]
                if tgt in SOLID or tgt == OOB:
                    out["blocked"].append((e, t, a))
                elif tgt != LAVA and not mob_at(g, pos[g] + dd):
                    out["move"].append((e, t, a))
            f_cell = pos[g] + d[g]
            if a == DO and face[g] == TREE and L["s_inv"][g, 0] <= 8:
                out["wood"].append((e, t, a))
            if a == DO and face[g] == GRASS and L["s_inv"][g, 5] <= 8 and not mob_at(g, f_cell):
                out["sapling"].append((e, t, a))
            if a == DO and face[g] == WATER and L["s_body"][g, 2] <= 8:
                out["drink"].append((e, t, a))
            if a == PLACE_TABLE and L["s_inv"][g, 0] >= 2 and face[g] not in SOLID and face[g] != OOB:
                out["table"].append((e, t, a))
    counts = {c: len(v) for c, v in out.items()}
    for c in out:
        if len(out[c]) > N_MAX:
            out[c] = [out[c][i] for i in rng.choice(len(out[c]), N_MAX, replace=False)]
    return out, counts, frames


def hud_means(L, enc):
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    res = {}
    for name, (cell, arr, col) in HUD.items():
        p = torch.cat([enc[e]["p"][:, cell].float() for e in range(len(enc))])
        v = np.concatenate([L["s_body" if arr == 0 else "s_inv"][s_off[e]:s_off[e + 1], col] for e in range(len(enc))])
        res[name] = {int(k): p[v == k].mean(0) for k in np.unique(v)}
    return res


def table_means(L, enc):
    """Mean real periphery token of a crafting table at each of the 4 cells next to the player (screen row, col)."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = {}
    for dr, dc in FC.DIRS[1:]:
        cell = (3 + dr) * 9 + 4 + dc
        p = torch.cat([enc[e]["p"][:, cell].float() for e in range(len(enc))])
        v = np.concatenate([L["s_view"][s_off[e]:s_off[e + 1], 3 + dr, 4 + dc] for e in range(len(enc))])
        out[cell] = p[v == TABLE].mean(0)
    return out


def functional(kind, rows, enc, L, s_off, hm, tm):
    """A [B, 81, 16], W [B, 81, 16] with f(y) = ((y - A) * W).sum."""
    A, W = torch.zeros(len(rows), 81, 16), torch.zeros(len(rows), 81, 16)
    for i, (e, t, a) in enumerate(rows):
        g = s_off[e] + t - 1
        prev = enc[e]["p"][t - 1].float()
        if kind in ("move", "blocked"):
            dd = FC.DIRS[a]
            for r in range(7):
                for c in range(9):
                    sr, sc = r + dd[0], c + dd[1]
                    if not (0 <= sr < 7 and 0 <= sc < 9) or (r, c) in ((3, 4), (3 - dd[0], 4 - dd[1])):
                        continue
                    A[i, r * 9 + c], W[i, r * 9 + c] = prev[r * 9 + c], prev[sr * 9 + sc] - prev[r * 9 + c]
        elif kind == "table":
            dd = FC.DIRS[L["s_dir"][g]]
            cell = (3 + dd[0]) * 9 + 4 + dd[1]
            A[i, cell], W[i, cell] = prev[cell], tm[cell] - prev[cell]
        else:
            cell, arr, col = HUD[kind]
            v = int(L["s_body" if arr == 0 else "s_inv"][g, col])
            A[i, cell], W[i, cell] = hm[kind][v], hm[kind][v + 1] - hm[kind][v]
    den = W.square().sum((-1, -2))
    W /= den.clamp(min=1e-9)[:, None, None]
    return A, W, den >= MIN_SEPARATION


def recorded_ok(kind, rows, L, s_off):
    ok = []
    for e, t, a in rows:
        g0, g1 = s_off[e] + t - 1, s_off[e] + t
        if kind == "move":
            ok.append((L["s_pos"][g1] - L["s_pos"][g0] == FC.DIRS[a]).all())
        elif kind == "blocked":
            ok.append((L["s_pos"][g1] == L["s_pos"][g0]).all())
        elif kind == "table":
            dd = FC.DIRS[L["s_dir"][g0]]
            ok.append(L["s_view"][g1, 3 + dd[0], 4 + dd[1]] == TABLE)
        else:
            cell, arr, col = HUD[kind]
            x = L["s_body" if arr == 0 else "s_inv"]
            ok.append(x[g1, col] == x[g0, col] + 1)
    return np.array(ok, bool)


@torch.no_grad()
def run(model, enc, L, rows, kind, hm, tm, cfg, gen, heun=True, keep=False):
    """keep (added 10-30, fovea fine-tune evaluation): also return the production samples (gen_p, gen_g, gen_v), same
    draws."""
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = {k: [] for k in ("f_real", "f_den", "f_gen", "f_heun", "row_ctx", "valid", "gen_p", "gen_g", "gen_v")}
    n_ctx = 256 // R
    for start in range(0, len(rows), n_ctx):
        batch = rows[start:start + n_ctx]
        A, W, valid = functional(kind, batch, enc, L, s_off, hm, tm)
        idx = [i for i in range(len(batch)) for _ in range(R)]
        A, W = A[idx].to(DEV), W[idx].to(DEV)
        out["valid"].append(valid[idx])
        f = lambda y: ((y.float() - A) * W).sum((-1, -2)).cpu()
        take = lambda key, dt: torch.stack([enc[batch[i][0]][key][batch[i][1] + dt] for i in idx]).to(DEV)
        B = len(idx)
        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-P, 0):
                x = tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v"))
                cache = M.context_pass(den, x, take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None],
                                       cache, cfg, gen)
            bp = take("bp", 0).float()
            pa, mem, fx = take("prev_action", 0)[:, None], model.belief.memory(bp)[:, None], take("fix", 0)[:, None]
            x0 = tuple(take(k, 0).float() for k in ("p", "g", "v"))
            out["f_real"].append(f(x0[0]))
            fd = []
            for s in SIGMAS:
                xs = tuple(t[:, None] + s * torch.randn(t[:, None].shape, device=DEV, generator=gen) for t in x0)
                D, _ = M.denoised(den, xs, torch.full((B, 1), s, device=DEV), pa, mem, fx, cache, clamp, sd)
                fd.append(f(D[0][:, 0]))
            out["f_den"].append(torch.stack(fd, 1))
            shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
            xg = M.sample(den, shapes, pa, mem, fx, cache, cfg, gen)
            out["f_gen"].append(f(xg[0][:, 0]))
            if keep:
                for k_, t_ in zip(("gen_p", "gen_g", "gen_v"), xg):
                    out[k_].append(t_[:, 0].half().cpu())
            if heun:
                out["f_heun"].append(f(SS.sample(den, shapes, pa, mem, fx, cache, SS.ALGORITHMS["heun_32"], cfg, gen)[0]))
        out["row_ctx"].append(torch.tensor([start + i for i in idx]))
    return {k: torch.cat(v).numpy() for k, v in out.items() if v}


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(FC.CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    enc = S.encode(model, L, cfg)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    ctx, counts, frames = contexts(L, enc, np.random.default_rng(2031))
    hm, tm = hud_means(L, enc), table_means(L, enc)
    for kind, (cell, arr, col) in HUD.items():                           # both counts must have real tokens
        x = L["s_body" if arr == 0 else "s_inv"]
        ctx[kind] = [r for r in ctx[kind] if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= set(hm[kind])]
    assert all(torch.isfinite(v).all() for v in tm.values())
    gen = torch.Generator(device=DEV).manual_seed(2031)
    res = {"checkpoint_sha256": TR.file_sha256(FC.CKPT), "sigmas": SIGMAS, "frames": frames,
           "opportunities": counts, "opportunity_rate": {c: n / frames for c, n in counts.items()}}
    for kind, rows in ctx.items():
        if not rows:
            continue
        o = run(model, enc, L, rows, kind, hm, tm, cfg, gen)
        np.savez_compressed(out_dir / f"{kind}.npz", **o, ctx=np.array(rows))
        ep = np.array([rows[i][0] for i in o["row_ctx"]])
        est = lambda v: G.estimate(np.asarray(v, float), o["valid"], ep)
        ok = recorded_ok(kind, rows, L, s_off)
        res[kind] = {"contexts": len(rows), "rows_valid": int(o["valid"].sum()), "recorded_as_defined": float(ok.mean()),
                     "f_recorded": est(o["f_real"]), "f_production": est(o["f_gen"]), "f_heun_32": est(o["f_heun"]),
                     "field_mean": {str(s): est(o["f_den"][:, k]) for k, s in enumerate(SIGMAS)},
                     "bias": {str(s): est(o["f_den"][:, k] - o["f_real"]) for k, s in enumerate(SIGMAS)}}
        r = res[kind]
        print(kind, len(rows), "ok", round(r["recorded_as_defined"], 4), "f rec", round(r["f_recorded"]["est"], 4),
              "field 0.7/2/6.67", [round(r["field_mean"][s]["est"], 4) for s in ("0.7", "2.0", "6.67")],
              "prod", round(r["f_production"]["est"], 4), "heun", round(r["f_heun_32"]["est"], 4),
              "bias 0.05", round(r["bias"]["0.05"]["est"], 4), flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
