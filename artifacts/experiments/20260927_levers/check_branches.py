"""Decision check (2026-10-02): why do longer-trained worlds make the imagined-state heads (dpanel gen1) WORSE next to zombies
(-0.034 / -0.036, both seeds) while real-state heads read their imagined states better (transfer1 +0.044 / +0.027) and zombie
probes on imagined states are no worse (teval zombie AUC 0.841 -> 0.857, 0.826 -> 0.854)? The gen head ranks the 17 actions
per root, so what matters is how the imagined successors DIFFER ACROSS ACTIONS.
dpanel judge rows (3,125 roots, opened blocks), H1 opportunity roots (P(death1) varies over actions), split zombie-adjacent /
other (dpanel's strata, from the rows file). Step-1 imagination as dpanel (4 context frames, branch action). Per root, over the
17 branches, tokens = near (the 3 x 3 around the player) or map (63):
  spread      mean over branches of |x_b - mean_b x|^2 (imagined, and real step-1 successors from dpanel_v1)
  alignment   cosine between the centred imagined [17, tokens x 192] and centred real matrices (does the imagined action-
              dependence point where the real one does?)
  error       mean over branches of |imagined_b - real_b|^2
  risk_rank   Spearman correlation across the 17 branches between real P(death1) and -(imagined distance of the branch from
              the NOOP branch near the player)... not used: reported only as the share of roots whose most dangerous real branch
              is also the imagined branch most different from NOOP near the player (top1_danger_match)
Reading, declared before running (each seed, zombie-adjacent roots, 36k vs 18k):
  branch_alignment_lost   near alignment lower by >= 0.05 at both seeds while error is not higher: the 36k imagined branches
                          differ from each other in the wrong directions near zombies
  branch_spread_lost      near spread ratio (imagined / real) lower by >= 20% at both seeds: less action-dependence drawn
Usage: check_branches.py <world.pt> ...
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import teval as T

N = 17
NEAR = [21, 22, 23, 30, 31, 32, 39, 40, 41]
D = "artifacts/eda/dpanel_v1/"
meta = torch.load(D + "judge_meta.pt", weights_only=False)
R = len(meta["seed"])
ctx = torch.from_numpy(np.memmap(D + "judge_ctx.f16", dtype=np.float16, mode="r", shape=(R, 4, 81, 192)))
real1 = torch.from_numpy(np.memmap(D + "judge_real1.f16", dtype=np.float16, mode="r", shape=(R, N, 81, 192)))
rows_ref = torch.load("artifacts/experiments/20260927_levers/evals/dpanel_corrt_raw_teacher_s7_u18000_rows.pt", weights_only=False)
assert torch.equal(rows_ref["seed"], meta["seed"])
opp, zomb = rows_ref["opp1"], rows_ref["zombie"]
dev = torch.device("cuda")
from d4mj.config import config_from_dict
import spatial as Sp
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])


def stats(img, real, idx):
    out = {}
    for name, tok in (("near", NEAR), ("map", list(range(63)))):
        a = img[:, :, tok].float().flatten(2); b = real[:, :, tok].float().flatten(2)
        ac, bc = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
        sp_i, sp_r = (ac ** 2).sum(-1).mean(1), (bc ** 2).sum(-1).mean(1)
        cos = (ac.flatten(1) * bc.flatten(1)).sum(1) / (ac.flatten(1).norm(dim=1) * bc.flatten(1).norm(dim=1)).clamp(min=1e-9)
        err = ((a - b) ** 2).sum(-1).mean(1)
        out[name] = {"spread_img": sp_i, "spread_real": sp_r, "alignment": cos, "error": err}
    p = meta["p1"][idx]
    noop_dist = ((img[:, :, NEAR].float() - img[:, :1, NEAR].float()) ** 2).sum((-1, -2))
    out["top1_danger_match"] = (p.argmax(1) == noop_dist.argmax(1)).float()
    return out


res = {}
for path in sys.argv[1:]:
    world, st = T.load_world(path, dev)
    acc = {}
    with torch.no_grad():
        for i in range(0, R, 16):
            idx = torch.arange(i, min(i + 16, R)); b = len(idx)
            fan = ctx[idx].float().repeat_interleave(N, 0)
            past = meta["acts"][idx, 1:].repeat_interleave(N, 0)
            a = torch.arange(N).repeat(b)[:, None]
            one = T.step(world, fan, torch.cat([past, a], 1), dev, config).view(b, N, 81, 192)
            s = stats(one, real1[idx].float(), idx)
            for k, v in s.items():
                if isinstance(v, dict):
                    for q, x in v.items():
                        acc.setdefault((k, q), []).append(x)
                else:
                    acc.setdefault((k,), []).append(v)
    cat = {k: torch.cat(v) for k, v in acc.items()}
    r = {}
    for stratum, m in (("zombie", opp & zomb), ("other", opp & ~zomb)):
        r[stratum] = {"roots": int(m.sum())}
        for tk in ("near", "map"):
            r[stratum][tk] = {"spread_ratio": float(cat[(tk, "spread_img")][m].mean() / cat[(tk, "spread_real")][m].mean()),
                              "alignment": float(cat[(tk, "alignment")][m].mean()), "error": float(cat[(tk, "error")][m].mean())}
        r[stratum]["top1_danger_match"] = float(cat[("top1_danger_match",)][m].mean())
    res[st["name"]] = r
    print(json.dumps({st["name"]: r}), flush=True)
    del world; torch.cuda.empty_cache()
rd = {}
for s in (7, 8):
    a, b = res.get(f"corrt_raw_teacher_s{s}_u18000"), res.get(f"corrt_raw_teacher_s{s}_u36000")
    if a and b:
        za, zb = a["zombie"]["near"], b["zombie"]["near"]
        rd[f"s{s}"] = {"alignment_drop": za["alignment"] - zb["alignment"], "error_change": zb["error"] - za["error"],
                       "spread_ratio_change": zb["spread_ratio"] / za["spread_ratio"] - 1}
if len(rd) == 2:
    rd["branch_alignment_lost"] = all(rd[s]["alignment_drop"] >= 0.05 and rd[s]["error_change"] <= 0 for s in ("s7", "s8"))
    rd["branch_spread_lost"] = all(rd[s]["spread_ratio_change"] <= -0.2 for s in ("s7", "s8"))
res["readings"] = rd
print(json.dumps(res))
