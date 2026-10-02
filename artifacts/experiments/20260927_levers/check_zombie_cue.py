"""Item 1 check (2026-10-03): is the 36k worlds' gen1 regression next to zombies (check_headseeds: -0.030 s7, -0.035 s8, both
resolved) a geometric cue lost by becoming MORE faithful? Craftax-Classic source (check_damage_rule): an attacking zombie stays put;
one that does not attack moves, 75% of the time toward the player. So in REAL successors a zombie often ends beside the player after
a SAFE action too (it chases), while a world that keeps the zombie in place makes "zombie beside the player after the step" the
attack condition itself. The 36k imagined branches are closer to the real ones (check_branches 0.928 -> 0.939, 0.927 -> 0.939).
dpanel judge (opened blocks v6 + v7), H1 opportunity roots, zombie-adjacent vs other; step-1 imagination as dpanel. Per branch, the
zombie score of teval's per-cell zombie probe (Probes.zombie), max over the 4 cells beside the player (22, 30, 32, 40) = "beside";
and the same max over the root's zombie cells as drawn after the step (zombie cells of the root's visible state, shifted by the
branch's REAL view shift) = "stayed". Per root, across the 17 branches: Pearson correlation of each with P(death1), averaged over
roots with variation; real successors and each world.
Readings, declared before running:
  beside_cue_lost   on zombie roots the 36k world's beside-correlation is lower than the 18k world's by >= 0.05 at both seeds
  real_cue_weaker   the real successors' beside-correlation is lower than every 18k world's
Added after a smoke run on the REAL successors only (zombie roots: beside 0.066, stayed 0.615), before any world was run:
  stayed_cue_lost   on zombie roots the 36k world's stayed-correlation is lower than the 18k world's by >= 0.05 at both seeds
Usage: check_zombie_cue.py <world.pt> ...
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import teval as T  # noqa: E402

N = 17
BESIDE = [22, 30, 32, 40]
D = "artifacts/eda/dpanel_v1/"
meta = torch.load(D + "judge_meta.pt", weights_only=False)
R = len(meta["seed"])
ctx = torch.from_numpy(np.memmap(D + "judge_ctx.f16", dtype=np.float16, mode="r", shape=(R, 4, 81, 192)))
real1 = torch.from_numpy(np.memmap(D + "judge_real1.f16", dtype=np.float16, mode="r", shape=(R, N, 81, 192)))
rows = torch.load("artifacts/experiments/20260927_levers/evals/dpanel_corrt_raw_teacher_s7_u18000_rows.pt", weights_only=False)
opp, zomb = rows["opp1"], rows["zombie"]
dev = torch.device("cuda")
from d4mj.config import config_from_dict  # noqa: E402
import spatial as Sp  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
tm, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), tm, tr, ts)
root_z = meta["visible"][:, 1071:1512].float().reshape(R, 7, 9, 7)[..., 0].flatten(1) > 0        # [R,63] zombie cells at the root
root_tok = ctx[:, -1]
shift = torch.cat([estimate(root_tok[i:i + 16].float()[:, None].expand(-1, N, -1, -1).flatten(0, 1),
                            real1[i:i + 16].float().flatten(0, 1)).view(-1, N) for i in range(0, R, 16)])   # [R,17] real view shift


def cues(x):
    """x [R,17,81,192] -> beside [R,17], stayed [R,17] (zombie score max over the cells)."""
    z = torch.cat([probes.zombie(x[i:i + 32, :, :63].float().flatten(0, 2)).view(-1, N, 63) for i in range(0, R, 32)])
    beside = z[:, :, BESIDE].amax(-1)
    off = torch.tensor(SHIFTS)[shift]                                   # [R,17,2]: next[r, c] = root[r + dr, c + dc]
    stayed = torch.full((R, N), float("nan"))
    rr, cc = torch.arange(63) // 9, torch.arange(63) % 9
    for r in range(R):
        cells = torch.where(root_z[r])[0]
        if len(cells) == 0:
            continue
        for a in range(N):
            dr, dc = int(off[r, a, 0]), int(off[r, a, 1])
            nr, nc = rr[cells] - dr, cc[cells] - dc
            ok = (nr >= 0) & (nr < 7) & (nc >= 0) & (nc < 9)
            if ok.any():
                stayed[r, a] = z[r, a, (nr * 9 + nc)[ok]].max()
    return beside, stayed


def corr(p, h):
    a = p - p.mean(1, keepdim=True); b = h - h.nanmean(1, keepdim=True)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 1e-9, (a * b).sum(1) / den.clamp(min=1e-9), torch.full_like(den, float("nan")))


def summarize(beside, stayed):
    out = {}
    for stratum, m in (("zombie", opp & zomb), ("other", opp & ~zomb)):
        res = {"roots": int(m.sum())}
        for name, v in (("beside", beside), ("stayed", stayed)):
            ok_rows = m & ~torch.isnan(v).any(1)
            c = corr(meta["p1"][ok_rows], v[ok_rows])
            res[f"{name}_corr"] = float(c[~torch.isnan(c)].mean()); res[f"{name}_roots"] = int((~torch.isnan(c)).sum())
            res[f"{name}_mean_dead"] = float(v[ok_rows][meta["p1"][ok_rows] > 0.5].mean())
            res[f"{name}_mean_safe"] = float(v[ok_rows][meta["p1"][ok_rows] < 0.5].mean())
        out[stratum] = res
    return out


def main():
    res = {"real": summarize(*cues(real1))}
    print(json.dumps({"real": res["real"]}), flush=True)
    for path in sys.argv[1:]:
        world, st = T.load_world(path, dev)
        img = torch.empty(R, N, 81, 192, dtype=torch.float16)
        with torch.no_grad():
            for i in range(0, R, 16):
                idx = torch.arange(i, min(i + 16, R)); b = len(idx)
                fan = ctx[idx].float().repeat_interleave(N, 0)
                past = meta["acts"][idx, 1:].repeat_interleave(N, 0)
                a = torch.arange(N).repeat(b)[:, None]
                img[idx] = T.step(world, fan, torch.cat([past, a], 1), dev, config).view(b, N, 81, 192).half()
        res[st["name"]] = summarize(*cues(img))
        print(json.dumps({st["name"]: res[st["name"]]}), flush=True)
        del world, img; torch.cuda.empty_cache()
    rd = {}
    for s in (7, 8):
        a, b = res.get(f"corrt_raw_teacher_s{s}_u18000"), res.get(f"corrt_raw_teacher_s{s}_u36000")
        if a and b:
            rd[f"s{s}_beside_change"] = b["zombie"]["beside_corr"] - a["zombie"]["beside_corr"]
            rd[f"s{s}_stayed_change"] = b["zombie"]["stayed_corr"] - a["zombie"]["stayed_corr"]
    if len(rd) == 4:
        rd["beside_cue_lost"] = all(rd[f"s{s}_beside_change"] <= -0.05 for s in (7, 8))
        rd["stayed_cue_lost"] = all(rd[f"s{s}_stayed_change"] <= -0.05 for s in (7, 8))
    w18 = [res[k] for k in res if k.endswith("u18000")]
    if w18:
        rd["real_cue_weaker"] = all(res["real"]["zombie"]["beside_corr"] < w["zombie"]["beside_corr"] for w in w18)
    res["readings"] = rd
    print(json.dumps(res))


if __name__ == "__main__":
    main()
