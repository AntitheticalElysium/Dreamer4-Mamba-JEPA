"""Item 1 check (2026-10-02): does the imagined HEALTH carry the action-dependent danger the gen heads learn from? (check_headseeds:
the 36k worlds' gen1 regression next to zombies is real (-0.030 / -0.035); check_branches, teval zombie AUC and stochdiag say
their imagined step-1 states are closer to reality; the real-fitted heads read them better (transfer1 zombie +0.064 / +0.025).
A head fitted on imagined states with TRUE labels uses whatever imagined feature tracks danger; dying in one step next to a
zombie means health reaching 0.)
dpanel judge roots (opened blocks), H1 opportunity roots, zombie-adjacent vs other. Step-1 imagination as dpanel; health read by
teval's HUD ridge probe (Probes.hud column 0) on the imagined and the REAL step-1 successors. Per root, across the 17 branches:
Pearson correlation between P(death1) and -(health) ("danger_corr"; real successors = the ceiling), and the across-branch spread
of the health readout. Averaged over roots with non-constant P and health.
Reading, declared before running: health_signal_lost = on zombie roots the 36k world's danger_corr is lower than the 18k world's
by >= 0.05 at both seeds (the imagined health no longer tracks the dangerous action as strongly).
Usage: check_hud_danger.py <world.pt> ...
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import teval as T

N = 17
D = "artifacts/eda/dpanel_v1/"
meta = torch.load(D + "judge_meta.pt", weights_only=False)
R = len(meta["seed"])
ctx = torch.from_numpy(np.memmap(D + "judge_ctx.f16", dtype=np.float16, mode="r", shape=(R, 4, 81, 192)))
real1 = torch.from_numpy(np.memmap(D + "judge_real1.f16", dtype=np.float16, mode="r", shape=(R, N, 81, 192)))
rows = torch.load("artifacts/experiments/20260927_levers/evals/dpanel_corrt_raw_teacher_s7_u18000_rows.pt", weights_only=False)
opp, zomb = rows["opp1"], rows["zombie"]
dev = torch.device("cuda")
from d4mj.config import config_from_dict
import spatial as Sp
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
tm, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), tm, tr, ts)


def health(x):
    """[B,17,81,192] -> [B,17] health readout."""
    return probes.hud(x[:, :, 63:81].float().flatten(2).flatten(0, 1))[:, 0].view(x.shape[0], N)


def corr(p, h):
    """per-root Pearson(P(death1), -health) across branches; NaN where constant."""
    a = p - p.mean(1, keepdim=True); b = -h - (-h).mean(1, keepdim=True)
    den = a.norm(dim=1) * b.norm(dim=1)
    return torch.where(den > 1e-9, (a * b).sum(1) / den.clamp(min=1e-9), torch.full_like(den, float("nan")))


def summarize(c, spread):
    out = {}
    for stratum, m in (("zombie", opp & zomb), ("other", opp & ~zomb)):
        cm = c[m]; ok = ~torch.isnan(cm)
        out[stratum] = {"roots": int(m.sum()), "with_variation": int(ok.sum()), "danger_corr": float(cm[ok].mean()),
                        "health_spread": float(spread[m].mean())}
    return out


res = {}
p1 = meta["p1"]
h_real = torch.cat([health(real1[i:i + 64]) for i in range(0, R, 64)])
res["real_successors"] = summarize(corr(p1, h_real), h_real.std(1))
print(json.dumps({"real_successors": res["real_successors"]}), flush=True)
for path in sys.argv[1:]:
    world, st = T.load_world(path, dev)
    hs = []
    with torch.no_grad():
        for i in range(0, R, 16):
            idx = torch.arange(i, min(i + 16, R)); b = len(idx)
            fan = ctx[idx].float().repeat_interleave(N, 0)
            past = meta["acts"][idx, 1:].repeat_interleave(N, 0)
            a = torch.arange(N).repeat(b)[:, None]
            one = T.step(world, fan, torch.cat([past, a], 1), dev, config).view(b, N, 81, 192)
            hs.append(health(one))
    h = torch.cat(hs)
    res[st["name"]] = summarize(corr(p1, h), h.std(1))
    print(json.dumps({st["name"]: res[st["name"]]}), flush=True)
    del world; torch.cuda.empty_cache()
rd = {}
for s in (7, 8):
    a, b = res.get(f"corrt_raw_teacher_s{s}_u18000"), res.get(f"corrt_raw_teacher_s{s}_u36000")
    if a and b:
        rd[f"s{s}_zombie_corr_change"] = b["zombie"]["danger_corr"] - a["zombie"]["danger_corr"]
if len(rd) == 2:
    rd["health_signal_lost"] = all(v <= -0.05 for v in rd.values())
res["readings"] = rd
print(json.dumps(res))
