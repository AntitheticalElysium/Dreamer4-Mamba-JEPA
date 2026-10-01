"""E11h. Where does known content drift accumulate? (E11c: decisions flip on target tiles holding known, drifted content, not on
tiles the player stood on; E11f: on steps without a scroll, content-unchanged cells barely drift at depth 1.) Conjecture: the drift
is added at SCROLL steps, where the copy head rebuilds every tile from a neighbour (soft mixture weights, imagined inputs).

Same rollouts as driftanat.py. Along each root's imagined rollout, for steps k >= 2 where the imagined offset is right at k-1 and k,
all five samples are alive and agree on the offset: each map cell at step k is matched to the SAME world cell at step k-1 (screen
(r, c) at k was (r + dr, c + dc) at k-1 for the true scroll (dr, dc)). Cells kept: observable at the root (inside the root view),
no mob in any sample at k-1 or k, same drawn content (tile class) at k-1 and k in sample 0. Increment = excess_k - excess_{k-1}
(stochdiag's excess, / V), per cell.
Reported: mean increment per cell on scroll steps vs no-scroll steps, overall and by depth; the same for the four move targets.
Reading, declared before running:
  scroll_driven  per-cell increment on scroll steps >= 3x that on no-scroll steps (all kept cells, depths 2-16 pooled)
--hard (added after E11i, reading declared before that run): the corr head decodes with ITC's argmax (world.hard_decode: each
tile copies exactly one candidate). Reading: mixing_diffusion if the pooled per-cell increment on no-scroll steps falls >= 3x
against the soft run of the same world (the soft head's repeated re-mixing of its own outputs is what drifts known content).
Light split (added after E11j: hard decoding INCREASES known-cell drift, so the true tokens of "unchanged" cells move; Craftax's
light level changes every step, 1 - |cos(pi (t/300 mod 1 + 0.3))|^3, game_logic.calculate_light_level): increments binned by
|change in light level| between k-1 and k (visible state, sample 0): < 0.001, 0.001-0.01, 0.01-0.03, >= 0.03. Reading, declared
before that run: lighting_driven if the pooled per-cell increment (all kept cells) in the < 0.001 bin is <= 1/3 of the >= 0.03 bin.
Sleep split (added after E11m: true tokens of unchanged cells move 0.0013 per cell on sleeping steps vs 0.00006 awake; Craftax renders
the map grayscale at half brightness while sleeping, renderer.py "Apply sleep"): increments split by whether the player sleeps at
k-1 or k (visible state, sample 0). Reading, declared before that run: sleep_driven if sleeping steps carry >= 50% of the summed
increment over all kept cells.
Usage: scrolldrift.py <world.pt> ... [--hard] -> evals/scrolldrift_<name>[__hard].json
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
TARGETS = (22, 30, 32, 40)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
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
    mobs = (vis[..., 1071:1512].reshape(R, S, H, 63, 7).sum(-1) > 0).any(1)                                # [R,16,63]
    tiles0 = vis[:, 0, :, :1071].reshape(R, H, 63, 17).argmax(-1)                                          # [R,16,63]
    rr = torch.arange(63) // 9; cc = torch.arange(63) % 9
    is_target = torch.isin(torch.arange(63), torch.tensor(TARGETS))
    out_dir = HERE / "evals"
    hard = "--hard" in sys.argv
    for path in [Path(p) for p in sys.argv[1:] if p != "--hard"]:
        world, st = T.load_world(path, device)
        world.hard_decode = hard
        name = st["name"] + ("__hard" if hard else "")
        batch = 16 if world.backbone_kind == "full" else 4
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        for i in range(0, R, batch):
            b = min(batch, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
            for k in range(H):
                w = 4 if k == 0 else 5
                g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1), device, config)
                gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fk[:, k])
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_off = DA.offsets(torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)]))
        good = (img_off == true_off[:, 0]).all(-1) & alive & agree                                         # [R,16]
        acc = {(kind, grp): torch.zeros(2, H) for kind in ("scroll", "still") for grp in ("all", "targets")}
        edges = (0.001, 0.01, 0.03)
        light_acc = torch.zeros(2, len(edges) + 1)
        sleep_acc = torch.zeros(2, 2)                     # [sum, cells] x [awake, sleeping]
        for i in range(0, R, 16):
            b = min(16, R - i)
            x = fut5[i:i + b].float(); mu = x.mean(1)
            s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
            ex = (((gen[i:i + b].float() - mu) ** 2).sum(-1) - s2 / S)[..., :63] / V                        # [b,16,63]
            for j in range(b):
                r = i + j
                for k in range(1, H):
                    if not (good[r, k] and good[r, k - 1]):
                        continue
                    dr, dc = SHIFTS[int(true_shift[r, 0, k])]
                    pr, pc = rr + dr, cc + dc                                                              # same world cell at k-1
                    inside = (pr >= 0) & (pr < 7) & (pc >= 0) & (pc < 9)
                    prev_idx = (pr.clamp(0, 6) * 9 + pc.clamp(0, 8))
                    wr, wc = rr + int(true_off[r, 0, k, 0]), cc + int(true_off[r, 0, k, 1])
                    observable = (wr >= 0) & (wr < 7) & (wc >= 0) & (wc < 9)
                    keep = inside & observable & ~mobs[r, k] & ~mobs[r, k - 1][prev_idx] & (tiles0[r, k] == tiles0[r, k - 1][prev_idx])
                    inc = ex[j, k] - ex[j, k - 1][prev_idx]
                    kind = "scroll" if int(true_shift[r, 0, k]) != 0 else "still"
                    for grp, gm in (("all", keep), ("targets", keep & is_target)):
                        acc[(kind, grp)][0, k] += float(inc[gm].sum()); acc[(kind, grp)][1, k] += float(gm.sum())
                    dl = abs(float(vis[r, 0, k, 1521]) - float(vis[r, 0, k - 1, 1521]))
                    lb = sum(dl >= e for e in edges)
                    light_acc[0, lb] += float(inc[keep].sum()); light_acc[1, lb] += float(keep.sum())
                    sl = int(float(vis[r, 0, k, 1520]) > 0.5 or float(vis[r, 0, k - 1, 1520]) > 0.5)
                    sleep_acc[0, sl] += float(inc[keep].sum()); sleep_acc[1, sl] += float(keep.sum())
        per = {f"{kind}_{grp}": {"per_cell_by_depth": (v[0] / v[1].clamp(min=1)).tolist(), "cells_by_depth": v[1].tolist(),
                                  "pooled_per_cell": float(v[0].sum() / v[1].sum().clamp(min=1)), "cells": float(v[1].sum())}
               for (kind, grp), v in acc.items()}
        light = {f"bin{b}": {"per_cell": float(light_acc[0, b] / light_acc[1, b].clamp(min=1)), "cells": float(light_acc[1, b])}
                 for b in range(len(edges) + 1)}
        res = {"world": name, "increments": per, "light_bins": {"edges": edges, **light},
               "sleep": {lab: {"per_cell": float(sleep_acc[0, i] / sleep_acc[1, i].clamp(min=1)), "sum": float(sleep_acc[0, i]),
                               "cells": float(sleep_acc[1, i])} for i, lab in enumerate(("awake", "sleeping"))}}
        ratio = per["scroll_all"]["pooled_per_cell"] / max(per["still_all"]["pooled_per_cell"], 1e-12)
        res["readings"] = {"scroll_over_still": ratio, "scroll_driven": ratio >= 3,
                           "lighting_driven": light["bin0"]["per_cell"] <= light[f"bin{len(edges)}"]["per_cell"] / 3,
                           "sleep_share_of_increment": float(sleep_acc[0, 1] / sleep_acc[0].sum().clamp(min=1e-12)),
                           "sleep_driven": bool(sleep_acc[0, 1] >= 0.5 * sleep_acc[0].sum())}
        (out_dir / f"scrolldrift_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, **{k: round(v["pooled_per_cell"], 6) for k, v in per.items()},
                          "light": {b: (round(v["per_cell"], 6), int(v["cells"])) for b, v in light.items()}, **res["readings"]}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
