"""Decision check (2026-10-02): what does a real-fitted head miss in imagined states? (check_hud_danger: the real step-1 health tracks
which action is deadly (per-root correlation with P(death1) 0.91 / 0.93, health spread across branches 0.05-0.10); every world's
imagined health ignores the action (correlation ~0, spread 0.010-0.013) -- imagination never draws the damage. dpanel transfer1
(real heads on imagined states) 0.53-0.69 vs real1 0.999.)
dpanel judge roots (opened blocks), H1 opportunity roots. dpanel's own real1 heads (3 seeds, cached under dpanel_v1/heads), read
on step-1 states built by substitution from the world's imagined state and the REAL successor:
  imagined          as dpanel's transfer1
  +real_hud         the 18 HUD tokens (63-80) taken from the real successor
  +real_near        the 3 x 3 around the player taken from the real successor
  +real_hud_near    both
  real              the real successor (dpanel's real1)
Expected safe per root (seed mean), overall and zombie-adjacent.
Reading, declared before running: hud_bottleneck = (+real_hud - imagined) >= 50% of (real - imagined) overall at every world: the
missing damage / HUD rendering is the main thing a real-fitted head cannot read in imagination.
Usage: check_transfer_subst.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import dpanel as D
T = D.T
N = 17
NEAR = [21, 22, 23, 30, 31, 32, 39, 40, 41]
C = D.CACHE
meta = torch.load(C / "judge_meta.pt", weights_only=False)
R = len(meta["seed"])
ctx = torch.from_numpy(np.memmap(C / "judge_ctx.f16", dtype=np.float16, mode="r", shape=(R, 4, 81, 192)))
real1 = torch.from_numpy(np.memmap(C / "judge_real1.f16", dtype=np.float16, mode="r", shape=(R, N, 81, 192)))
rows_ref = torch.load("artifacts/experiments/20260927_levers/evals/dpanel_corrt_raw_teacher_s7_u18000_rows.pt", weights_only=False)
opp, zomb = rows_ref["opp1"], rows_ref["zombie"]
dev = torch.device("cuda")
from d4mj.config import config_from_dict
from observability import expected_safe
import spatial as Sp
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
heads = [D.fit_head(real1, None, None, None, dev, s, C / "heads" / f"real1_s{s}.pt")[0] for s in range(3)]
rows_j = torch.arange(R)


def safe(x):
    runs = [expected_safe(D.judge_risk(m, x, rows_j, dev), meta["p1"])[0] for m in heads]
    s = torch.stack(runs).mean(0)
    return {"overall": float(s[opp].mean()), "zombie": float(s[opp & zomb].mean())}


res = {"real": safe(real1)}
print(json.dumps({"real": res["real"]}), flush=True)
for path in sys.argv[1:]:
    world, st = T.load_world(Path(path), dev)
    img = torch.empty(R, N, 81, 192, dtype=torch.float16)
    with torch.no_grad():
        for i in range(0, R, 16):
            idx = torch.arange(i, min(i + 16, R)); b = len(idx)
            fan = ctx[idx].float().repeat_interleave(N, 0)
            past = meta["acts"][idx, 1:].repeat_interleave(N, 0)
            a = torch.arange(N).repeat(b)[:, None]
            img[idx] = T.step(world, fan, torch.cat([past, a], 1), dev, config).view(b, N, 81, 192).half()
    r = {"imagined": safe(img)}
    for name, cells in (("+real_hud", list(range(63, 81))), ("+real_near", NEAR), ("+real_hud_near", list(range(63, 81)) + NEAR)):
        x = img.clone(); x[:, :, cells] = real1[:, :, cells]
        r[name] = safe(x)
    gap = res["real"]["overall"] - r["imagined"]["overall"]
    r["hud_share_of_gap"] = (r["+real_hud"]["overall"] - r["imagined"]["overall"]) / gap if gap > 0 else None
    r["near_share_of_gap"] = (r["+real_near"]["overall"] - r["imagined"]["overall"]) / gap if gap > 0 else None
    res[st["name"]] = r
    print(json.dumps({st["name"]: r}), flush=True)
    del world, img; torch.cuda.empty_cache()
ws = [k for k in res if k != "real"]
res["readings"] = {"hud_bottleneck": all((res[k]["hud_share_of_gap"] or 0) >= 0.5 for k in ws)}
print(json.dumps(res))
