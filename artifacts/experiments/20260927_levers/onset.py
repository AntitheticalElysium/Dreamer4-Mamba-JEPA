"""E11o. When and how did the decision-flipping target tile get corrupted? (E11c: at the first wrong move decision the target tile's
imagined token is far from the truth, squared distance 49-77 vs 3-18 on correct decisions; single-factor tests -- mobs, occupied,
revealed, sleep, faced tile, scroll copy, contextual change -- did not explain it.) Data-driven attribution.

Same rollouts and cases as missedscroll.py (first wrong step of each root, missed or false scroll, depth >= 2; controls matched by
depth, up to 3 per case, seeded). For each case (root r, step k) the target tile's WORLD cell is followed back through the imagined
frames 0..k-1 (sample 0's true offsets; the imagined offset was right until k by construction). At each frame t where the cell is
in view, its excess e_t = |g_t - mu_t|^2 - s2_t/5 (/ V). The onset is the frame with the largest single increase e_t - e_{t-1}
(e_{-1} = 0 if the cell entered at t). At the onset, recorded: did the cell ENTER the view at t; was it the cell the player LEFT at t;
the FACED cell at t-1 under a DO / place action; did its drawn tile class CHANGE between t-1 and t; was a MOB within one cell of it
at t-1 or t (any sample); was the player ASLEEP at t-1 or t; did the view SCROLL at t. Also the share of the final error added at the
onset step.
Reported only (exploratory): event shares for wrong cases vs controls, and the onset depth relative to k. Joint split (added after
the first run): faced-tile DO / place events with the drawn tile class changed (a real consequence mispredicted) vs unchanged (a
consequence hallucinated), and the action at the onset step.
Usage: onset.py <world.pt> ... -> evals/onset_<name>.json
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
TARGET = {1: 30, 2: 32, 3: 22, 4: 40}
FACING_TILE = {0: (3, 3), 1: (3, 5), 2: (2, 4), 3: (4, 4)}


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
    shifts = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        shifts.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(shifts, 1)
    true_off = DA.offsets(true_shift).long()
    agree = (true_off == true_off[:, :1]).all(-1).all(1)
    valid = alive & agree
    vis = meta["future_visible"].float()
    rootv = meta["root_visible"].float()
    allv = torch.cat([rootv[:, None, None].expand(R, S, 1, 1534), vis], 2)                                # [R,5,17,..] index 0 = root
    tiles = allv[:, 0, :, :1071].reshape(R, H + 1, 7, 9, 17).argmax(-1)                                     # [R,17,7,9]
    mobs = (allv[..., 1071:1512].reshape(R, S, H + 1, 7, 9, 7).sum(-1) > 0).any(1)                          # [R,17,7,9]
    asleep = allv[:, 0, :, 1520] > 0.5                                                                      # [R,17]
    facing = allv[:, 0, :, 1516:1520].argmax(-1)                                                            # [R,17]
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
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
        img_shift = torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)])
        aligned = (DA.offsets(img_shift).long() == true_off[:, 0]).all(-1)
        prev_ok = torch.cat([torch.ones(R, 1, dtype=torch.bool), aligned[:, :-1]], 1)
        move = (fa >= 1) & (fa <= 4)
        wrong = valid & prev_ok & ~aligned & move & (((true_shift[:, 0] != 0) & (img_shift == 0)) | ((true_shift[:, 0] == 0) & (img_shift != 0)))
        first = wrong & (torch.cumsum((valid & ~aligned).int(), 1) == 1)
        first[:, 0] = False
        right = valid & prev_ok & aligned & move & (img_shift == true_shift[:, 0])
        rng = torch.Generator().manual_seed(20261001)
        cases = [(int(r), int(k)) for r, k in torch.nonzero(first)]
        controls = []
        for k in sorted({k for _, k in cases}):
            n_k = sum(1 for _, kk in cases if kk == k)
            pool = torch.nonzero(right[:, k])[:, 0]
            controls += [(int(r), k) for r in pool[torch.randperm(len(pool), generator=rng)[:3 * n_k]]]
        excess_cache = {}

        def excess(r, t):                                                       # [63] excess of imagined frame t (future index)
            if (r, t) not in excess_cache:
                x = fut5[r, :, t].float(); mu = x.mean(0)
                s2 = ((x - mu) ** 2).sum((0, -1)) / (S - 1)
                excess_cache[(r, t)] = ((((gen[r, t].float() - mu) ** 2).sum(-1) - s2 / S) / V)[:63]
            return excess_cache[(r, t)]

        def attribute(r, k):
            tr_, tc_ = divmod(TARGET[int(fa[r, k])], 9)
            off_cur = true_off[r, 0, k - 1]
            wr, wc = tr_ + int(off_cur[0]), tc_ + int(off_cur[1])                # root-relative world cell
            series = []                                                          # (t, screen r, screen c, excess)
            for t in range(k):                                                   # imagined frames 0..k-1
                o = true_off[r, 0, t]
                sr, sc = wr - int(o[0]), wc - int(o[1])
                if 0 <= sr < 7 and 0 <= sc < 9:
                    series.append((t, sr, sc, float(excess(r, t)[sr * 9 + sc])))
            if not series:
                return None
            in_root = 0 <= wr < 7 and 0 <= wc < 9                               # the root frame is true: error 0 there
            best, prev_e, prev_t = None, 0.0, (-1 if in_root else None)
            for t, sr, sc, e in series:
                inc = e - (prev_e if prev_t == t - 1 else 0.0)
                if best is None or inc > best[0]:
                    best = (inc, t, sr, sc, prev_t == t - 1)
                prev_e, prev_t = e, t
            inc, t, sr, sc, was_in_view = best
            ti = t + 1                                                           # index into allv / tiles (0 = root)
            a_t = int(fa[r, t])
            fr, fc = FACING_TILE[int(facing[r, ti - 1])]
            ps = SHIFTS[int(true_shift[r, 0, t])]
            near = mobs[r, ti - 1, max(0, sr - 1):sr + 2, max(0, sc - 1):sc + 2].any() | mobs[r, ti, max(0, sr - 1):sr + 2, max(0, sc - 1):sc + 2].any()
            prev_sr, prev_sc = sr + ps[0], sc + ps[1]
            changed = (0 <= prev_sr < 7 and 0 <= prev_sc < 9) and bool(tiles[r, ti, sr, sc] != tiles[r, ti - 1, prev_sr, prev_sc])
            return {"onset_depth_before_k": k - t, "share_of_final": inc / max(series[-1][3], 1e-12), "action": a_t,
                    "entered": not was_in_view, "left_by_player": ps != (0, 0) and (sr, sc) == (3 - ps[0], 4 - ps[1]),
                    "faced_act": (prev_sr, prev_sc) == (fr, fc) and a_t in (5, 7, 8, 9, 10),
                    "tile_changed": changed, "mob_near": bool(near),
                    "asleep": bool(asleep[r, ti] or asleep[r, ti - 1]), "scrolled": ps != (0, 0)}
        res = {"world": name, "n": {"wrong": len(cases), "control": len(controls)}}
        for label, lst in (("wrong", cases), ("control", controls)):
            recs = [x for x in (attribute(r, k) for r, k in lst) if x is not None]
            keys = ("entered", "left_by_player", "faced_act", "tile_changed", "mob_near", "asleep", "scrolled")
            res[label] = {key: sum(x[key] for x in recs) / max(len(recs), 1) for key in keys}
            res[label]["n_attributed"] = len(recs)
            n = max(len(recs), 1)
            res[label]["faced_act_and_changed"] = sum(x["faced_act"] and x["tile_changed"] for x in recs) / n
            res[label]["faced_act_unchanged"] = sum(x["faced_act"] and not x["tile_changed"] for x in recs) / n
            res[label]["action_at_onset"] = {str(a): sum(x["action"] == a for x in recs) / n for a in range(17)
                                            if any(x["action"] == a for x in recs)}
            res[label]["onset_depth_before_k_mean"] = sum(x["onset_depth_before_k"] for x in recs) / max(len(recs), 1)
            res[label]["share_of_final_median"] = float(torch.tensor([x["share_of_final"] for x in recs]).median()) if recs else None
        (out_dir / f"onset_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
