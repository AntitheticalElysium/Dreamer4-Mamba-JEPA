"""E11i. Does the copy head lose confidence on its own outputs? (E11h: known, unchanged cells gain ~1.5-3e-4 excess per step once the
input is imagined, scroll or not; from true frames the same cells gain ~1-4e-5 (E11f). The corr heads output a softmax mixture of
{self, up, down, left, right, generate} per tile (tworld.TWorld.forward, `last_weights`); a weight of 1 on the correct source
copies exactly, so drift on unchanged cells requires that weight to fall.)

Same rollouts and cell selection as scrolldrift.py: steps k >= 2 with the imagined offset right at k-1 and k, all five samples
alive and agreeing; map cells observable at the root, no mob, same drawn content at k-1 and k. The correct source of each cell is
`self` (no scroll) or the neighbour matching the true scroll. At each such step the head is read twice: on the window the imagined
rollout used, and on the TRUE window at the same positions (sample 0's frames).
Reported: mean weight on the correct source and on `generate`, imagined vs true, overall and by depth; the share of cells with
correct-source weight < 0.9; Spearman correlation between (1 - imagined correct weight) and the cell's excess increment.
Reading, declared before running:
  copy_confidence_drop  imagined correct-source weight <= true correct-source weight - 0.05 (pooled), AND Spearman >= 0.3
Usage: copyconf.py <world.pt> ... -> evals/copyconf_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import driftanat as DA  # noqa: E402
SD, T = DA.SD, DA.T
H, S = SD.H, SD.S


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp(min=1e-12))


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    from scroll import SHIFTS, estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    true_shift = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        true_shift.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(true_shift, 1)
    true_off = DA.offsets(true_shift)
    agree = (true_off == true_off[:, :1]).all(-1).all(1)
    vis = meta["future_visible"].float()
    mobs = (vis[..., 1071:1512].reshape(R, S, H, 63, 7).sum(-1) > 0).any(1)
    tiles0 = vis[:, 0, :, :1071].reshape(R, H, 63, 17).argmax(-1)
    rr = torch.arange(63) // 9; cc = torch.arange(63) % 9
    fut0 = fut5[:, 0]
    out_dir = HERE / "evals"

    def run(world, frames, acts):
        with autocast_context(config):
            out = world(frames.to(device).float(), acts.to(device))[0]
        return out[:, -1].float().cpu(), world.last_weights[:, -1].float().cpu()          # [b,81,192], [b,81,6]

    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        if world.head not in ("corr", "corrg", "corrt"):
            print(json.dumps({"world": name, "skipped": "no copy mixture"}), flush=True)
            continue
        batch = 16 if world.backbone_kind == "full" else 4
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        w_img = torch.empty(R, H, 81, 6); w_true = torch.empty(R, H, 81, 6)
        for i in range(0, R, batch):
            b = min(batch, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
            tframes = [c4[:, j] for j in range(4)]
            for k in range(H):
                w = 4 if k == 0 else 5
                acts = torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                g, wi = run(world, torch.stack(frames[-w:], 1), acts)
                _, wt = run(world, torch.stack(tframes[-w:], 1), acts)
                gen[i:i + b, k] = g.half(); w_img[i:i + b, k] = wi; w_true[i:i + b, k] = wt
                frames.append(g); tframes.append(fut0[i:i + b, k].float()); hist.append(fk[:, k])
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_off = DA.offsets(torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)]))
        good = (img_off == true_off[:, 0]).all(-1) & alive & agree
        ci, ct, gi, gt, incs, lows = [], [], [], [], [], []
        by_depth = {k: [[], []] for k in range(1, H)}
        for i in range(0, R, 16):
            b = min(16, R - i)
            x = fut5[i:i + b].float(); mu = x.mean(1)
            s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
            ex = (((gen[i:i + b].float() - mu) ** 2).sum(-1) - s2 / S)[..., :63] / V
            for j in range(b):
                r = i + j
                for k in range(1, H):
                    if not (good[r, k] and good[r, k - 1]):
                        continue
                    sidx = int(true_shift[r, 0, k])
                    dr, dc = SHIFTS[sidx]
                    pr, pc = rr + dr, cc + dc
                    inside = (pr >= 0) & (pr < 7) & (pc >= 0) & (pc < 9)
                    prev_idx = pr.clamp(0, 6) * 9 + pc.clamp(0, 8)
                    wr, wc = rr + int(true_off[r, 0, k, 0]), cc + int(true_off[r, 0, k, 1])
                    observable = (wr >= 0) & (wr < 7) & (wc >= 0) & (wc < 9)
                    keep = inside & observable & ~mobs[r, k] & ~mobs[r, k - 1][prev_idx] & (tiles0[r, k] == tiles0[r, k - 1][prev_idx])
                    if not keep.any():
                        continue
                    # weights that produced frame k come from the step whose OUTPUT is frame k (index k)
                    wi_c = w_img[r, k, :63, sidx][keep]; wt_c = w_true[r, k, :63, sidx][keep]
                    ci.append(wi_c); ct.append(wt_c)
                    gi.append(w_img[r, k, :63, 5][keep]); gt.append(w_true[r, k, :63, 5][keep])
                    incs.append((ex[j, k] - ex[j, k - 1][prev_idx])[keep])
                    by_depth[k][0].append(wi_c); by_depth[k][1].append(wt_c)
        ci, ct, gi, gt, incs = (torch.cat(v) for v in (ci, ct, gi, gt, incs))
        res = {"world": name, "cells": len(ci),
               "correct_weight": {"imagined": float(ci.mean()), "true": float(ct.mean())},
               "generate_weight": {"imagined": float(gi.mean()), "true": float(gt.mean())},
               "share_correct_below_0.9": {"imagined": float((ci < 0.9).float().mean()), "true": float((ct < 0.9).float().mean())},
               "spearman_one_minus_w_vs_increment": spearman(1 - ci, incs),
               "by_depth": {k + 1: {"imagined": float(torch.cat(v[0]).mean()), "true": float(torch.cat(v[1]).mean())}
                            for k, v in by_depth.items() if v[0]}}
        res["readings"] = {"copy_confidence_drop": res["correct_weight"]["imagined"] <= res["correct_weight"]["true"] - 0.05
                           and res["spearman_one_minus_w_vs_increment"] >= 0.3}
        (out_dir / f"copyconf_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "by_depth"}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
