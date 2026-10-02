"""Item 9 check (2026-10-02): what causes the first wrong move decisions of self-fed imagination, by kind? (subst16 part B, s7:
18k 439 cases (381 missed / 58 false scrolls), 36k 293 (184 / 109); at 36k the target tile alone repairs 97% of FALSE scrolls but
only 14% of MISSED ones, everything-but-the-target 76%; the entering-cell oracle removes 51% of the 36k world's position failures.)
subst16's base rollout and rule exactly (diagnosis futures, sample 0's actions, teval's window; first wrong move decision at depth
>= 2 among valid depths). Per case, at the CURRENT imagined frame (the input of the wrong decision):
  target class   the imagined target tile (the cell the move enters) read by teval's tile probe, and the TRUE class from the
                 simulator; passable vs solid by craftax_classic constants.SOLID_BLOCKS (+ out of bounds), and a mob in the
                 true target cell (simulator)
  revealed       whether the target's world cell lay outside the root view on the true camera path (content the world never saw)
  missed-scroll group repair: the current frame with ONE group replaced by the true tokens -- player token, HUD, the 8-cell ring
                 around the player minus the target, every other map token -- and the drawn scroll compared with the truth
Readings, declared before running (per world):
  false_passable_drawn   among false scrolls, imagined target passable while truly solid or a mob >= 0.7
  false_on_revealed      among those, the target cell revealed after the root >= 0.5 (entering content drives false scrolls)
  missed_cause           the group repairing the most missed cases, with its rate (reported)
Usage: check_decision_step.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import subst16 as SB
DA, SD, T = SB.DA, SB.SD, SB.T
H = SD.H
TARGET = SB.TARGET
SOLID = {1, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16}            # out of bounds + craftax_classic constants.SOLID_BLOCKS (verified)
RING = [21, 22, 23, 30, 32, 39, 40, 41]


def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    shifts = []
    for s in range(5):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        shifts.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(shifts, 1); ts0 = true_shift[:, 0]
    true_off = DA.offsets(true_shift).long()
    valid = alive & (true_off == true_off[:, :1]).all(-1).all(1)
    allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()
    tiles_true = allv[:, :, :1071].reshape(R, 17, 63, 17).argmax(-1)                     # [R,17,63], 0 = root
    mobs_true = allv[:, :, 1071:1512].reshape(R, 17, 7, 9, 7).sum(-1).flatten(2) > 0      # [R,17,63]
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    fut0 = fut5[:, 0]
    out = {}
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        name = st["name"]
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        with torch.no_grad():
            for i in range(0, R, 64):
                b = min(64, R - i)
                c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
                frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
                for k in range(H):
                    w = 4 if k == 0 else 5
                    g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1), device, config)
                    gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fk[:, k])
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_shift = torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])
        aligned = (DA.offsets(img_shift).long() == true_off[:, 0]).all(-1)
        prev_ok = torch.cat([torch.ones(R, 1, dtype=torch.bool), aligned[:, :-1]], 1)
        move = (fa >= 1) & (fa <= 4)
        first = valid & prev_ok & ~aligned & move & (torch.cumsum((valid & ~aligned).int(), 1) == 1)
        first[:, 0] = False
        cases = torch.nonzero(first).tolist()
        recs = []
        groups = {"player": [31], "hud": list(range(63, 81)), "ring": None, "map_rest": None}
        for j in range(0, len(cases), 64):
            chunk = cases[j:j + 64]
            wins = torch.stack([torch.cat([ctx[r].float(), gen[r, :k].float()], 0)[-5:] for r, k in chunk])
            acts = torch.stack([torch.cat([ca[r], fa[r, :k + 1]], 0)[-5:] for r, k in chunk])
            cur_true = torch.stack([fut0[r, k - 1].float() for r, k in chunk])
            truth = torch.tensor([int(ts0[r, k]) for r, k in chunk])
            tg = [TARGET[int(fa[r, k])] for r, k in chunk]
            img_cls = probes.tile(torch.stack([wins[q, -1, tg[q]] for q in range(len(chunk))])).argmax(-1)
            repairs = {}
            for gname in groups:
                wv = wins.clone(); cur = wv[:, -1]
                for q in range(len(chunk)):
                    cells = groups[gname] if groups[gname] is not None else (
                        [c for c in RING if c != tg[q]] if gname == "ring" else [c for c in range(63) if c not in RING and c != 31 and c != tg[q]])
                    cur[q, cells] = cur_true[q, cells]
                o = T.step(world, wv, acts, device, config)
                repairs[gname] = (estimate(wv[:, -1].float(), o) == truth)
            for q, (r, k) in enumerate(chunk):
                t = tg[q]; row, col = divmod(t, 9)
                off = true_off[r, 0, k - 1]
                revealed = not (0 <= row + int(off[0]) < 7 and 0 <= col + int(off[1]) < 9)
                tc = int(tiles_true[r, k, t])
                recs.append({"kind": "missed" if int(ts0[r, k]) != 0 else "false", "img_class": int(img_cls[q]), "true_class": tc,
                             "img_passable": int(img_cls[q]) not in SOLID, "true_blocking": tc in SOLID or bool(mobs_true[r, k, t]),
                             "true_mob": bool(mobs_true[r, k, t]), "revealed": revealed,
                             **{f"repair_{g}": bool(repairs[g][q]) for g in groups}})
        f = [x for x in recs if x["kind"] == "false"]; m = [x for x in recs if x["kind"] == "missed"]
        share = lambda xs, key: round(sum(x[key] for x in xs) / max(len(xs), 1), 4)
        pd = [x for x in f if x["img_passable"] and x["true_blocking"]]
        res = {"cases": len(recs), "false": len(f), "missed": len(m),
               "false_img_passable_true_blocking": share(f, "img_passable") and round(len(pd) / max(len(f), 1), 4),
               "false_true_mob": share(f, "true_mob"), "false_revealed": share(f, "revealed"),
               "false_passable_drawn_revealed": share(pd, "revealed"),
               "false_true_classes": {c: sum(1 for x in f if x["true_class"] == c) for c in sorted({x["true_class"] for x in f})},
               "false_img_classes": {c: sum(1 for x in f if x["img_class"] == c) for c in sorted({x["img_class"] for x in f})},
               "missed_revealed": share(m, "revealed"),
               "missed_repair": {g: share(m, f"repair_{g}") for g in groups}, "false_repair": {g: share(f, f"repair_{g}") for g in groups}}
        res["readings"] = {"false_passable_drawn": res["false_img_passable_true_blocking"] >= 0.7,
                           "false_on_revealed": res["false_passable_drawn_revealed"] >= 0.5,
                           "missed_cause": max(res["missed_repair"], key=res["missed_repair"].get)}
        out[name] = res
        print(json.dumps({name: res}), flush=True)
        del world; torch.cuda.empty_cache()
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
