"""Item 9 follow-up (2026-10-02): HOW are obstacles erased before a false scroll, and WHICH map content blocks a scroll?
(check_decision_step: false scrolls draw a truly blocking target as passable (0.85-0.93) but the target was revealed after the
root in only 0.14-0.50 of them -- 36k: tables / furnaces drawn as grass; s8 18k: 120 stone targets drawn as path; missed
scrolls are repaired by the rest of the map 0.36-0.67.)
Same base rollout and first-wrong-move rule as check_decision_step. The imagined view is aligned until the wrong decision, so the
target's world cell has a known screen position at every earlier frame (true camera path).
(i) False scrolls with the target drawn passable but truly blocking, classified by the history of that world cell:
  placed_never_drawn   the truly blocking object was not there at the root (it appeared in reality during the rollout: a
                       placement) and the imagined cell was never drawn blocking
  entering_misdrawn    the cell entered the view during the rollout and was drawn passable from its first appearance
  erased               the cell was drawn blocking at some frame and passable later; recorded: the action of the step that
                       erased it, and whether that cell was the agent's FACED cell at that step (simulator facing)
  never_blocking_seen  visible at the root (or before entering) but never drawn blocking (a misdraw of known content)
(ii) Missed scrolls: the current frame with the REVEALED map cells (outside the root view on the true path) replaced by the true
tokens, or with the OBSERVABLE map cells (inside it; excluding the ring, the target and the player) replaced: scroll repair rates.
Reading, declared before running (per world): erasure_cause = the largest category of (i), and for `erased` the share at DO /
place steps on the faced cell (hallucinated consequence) vs other steps (drift); missed_region = the region repairing more.
Usage: check_erasure.py <world.pt> ...
Result (2026-10-02; s7 18k / s7 36k / s8 18k / s8 36k): false-passable cases 52 / 95 / 228 / 118.
  entering_misdrawn 26 / 42 / 31 / 38; erased 6 / 30 / 175 / 56 (erasing step DO on the faced cell 0 / 73 / 73 / 86%);
  placed_never_drawn 12 / 16 / 14 / 16; never_blocking_seen 8 / 7 / 8 / 8. erasure_cause: entering (s7) / erased (s8).
  Missed scrolls: revealed-region repair 0.30 / 0.58 / 0.57 / 0.60 vs observable 0.18 / 0.17 / 0.31 / 0.19 (missed_region revealed x4).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_decision_step as CD
SB, DA, SD, T = CD.SB, CD.DA, CD.SD, CD.T
H, TARGET, SOLID, RING = CD.H, CD.TARGET, CD.SOLID, CD.RING
FACED = {0: 30, 1: 32, 2: 22, 3: 40}                     # simulator facing index (left, right, up, down) -> screen cell
ACT = {5: "do", 6: "sleep", 7: "place", 8: "place", 9: "place", 10: "place", 0: "noop"}


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
    tiles_true = allv[:, :, :1071].reshape(R, 17, 63, 17).argmax(-1)
    mobs_true = allv[:, :, 1071:1512].reshape(R, 17, 7, 9, 7).sum(-1).flatten(2) > 0
    facing = allv[:, :, 1516:1520].argmax(-1)                                             # [R,17]
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
        cats = {"placed_never_drawn": 0, "entering_misdrawn": 0, "erased": 0, "never_blocking_seen": 0}
        erased_steps = []
        miss = {"revealed": [], "observable": []}
        for r, k in cases:
            t = TARGET[int(fa[r, k])]; row, col = divmod(t, 9)
            off_k = true_off[r, 0, k - 1]
            cur_img = int(probes.tile(gen[r, k - 1, t][None].float()).argmax(-1))
            tc = int(tiles_true[r, k, t])
            blocking = tc in SOLID or bool(mobs_true[r, k, t])
            if int(ts0[r, k]) == 0 and cur_img not in SOLID and blocking:     # a false scroll into a drawn-passable obstacle
                hist_img, hist_true = [], []
                for d in range(0, k + 1):                                       # frame d: 0 = root, d >= 1 = gen[r, d-1]
                    off_d = true_off[r, 0, d - 1] if d >= 1 else torch.zeros(2, dtype=torch.long)
                    rr, cc = row + int(off_k[0] - off_d[0]), col + int(off_k[1] - off_d[1])
                    if not (0 <= rr < 7 and 0 <= cc < 9):
                        hist_img.append(None); hist_true.append(None); continue
                    cell = rr * 9 + cc
                    tok = root[r, cell] if d == 0 else gen[r, d - 1, cell]
                    hist_img.append(int(probes.tile(tok[None].float()).argmax(-1)))
                    hist_true.append((int(tiles_true[r, d, cell]), cell))
                seen = [d for d in range(k + 1) if hist_img[d] is not None]
                true_at_root = hist_true[0][0] if hist_true[0] is not None else None
                drawn_blocking = [d for d in seen if hist_img[d] in SOLID]
                if not drawn_blocking:
                    if hist_true[0] is not None and true_at_root not in SOLID and tc in SOLID:
                        cats["placed_never_drawn"] += 1
                    elif hist_img[0] is None:
                        cats["entering_misdrawn"] += 1
                    else:
                        cats["never_blocking_seen"] += 1
                else:
                    cats["erased"] += 1
                    last_block = max(drawn_blocking)
                    d_erase = next(d for d in seen if d > last_block)
                    a = int(fa[r, d_erase - 1]) if d_erase >= 1 else -1
                    cell_prev = hist_true[d_erase - 1][1] if hist_true[d_erase - 1] is not None else None
                    faced = cell_prev is not None and FACED[int(facing[r, d_erase - 1])] == cell_prev
                    erased_steps.append({"action": ACT.get(a, "move" if 1 <= a <= 4 else "craft"), "faced": bool(faced)})
            elif int(ts0[r, k]) != 0:                                           # a missed scroll: region repairs
                win = torch.cat([ctx[r].float(), gen[r, :k].float()], 0)[-5:]
                acts = torch.cat([ca[r], fa[r, :k + 1]], 0)[-5:]
                cur_true = fut0[r, k - 1].float()
                rr_ = torch.arange(63) // 9; cc_ = torch.arange(63) % 9
                observable = ((rr_ + off_k[0]) >= 0) & ((rr_ + off_k[0]) < 7) & ((cc_ + off_k[1]) >= 0) & ((cc_ + off_k[1]) < 9)
                excl = torch.zeros(63, dtype=torch.bool); excl[RING + [t, 31]] = True
                for region, sel in (("revealed", ~observable & ~excl), ("observable", observable & ~excl)):
                    w2 = win.clone(); w2[-1, :63][sel] = cur_true[:63][sel]
                    o = T.step(world, w2[None], acts[None], device, config)
                    miss[region].append(bool(estimate(w2[-1:].float(), o) == int(ts0[r, k])))
        n_er = len(erased_steps)
        er = {"n": n_er}
        for key in ("do", "place", "move", "noop", "sleep", "craft"):
            er[key] = round(sum(1 for x in erased_steps if x["action"] == key) / max(n_er, 1), 4)
        er["at_do_or_place_on_faced"] = round(sum(1 for x in erased_steps if x["action"] in ("do", "place") and x["faced"]) / max(n_er, 1), 4)
        res = {"false_passable_cases": sum(cats.values()), "categories": cats, "erased_steps": er,
               "missed_cases": len(miss["revealed"]), "missed_repair": {k: round(sum(v) / max(len(v), 1), 4) for k, v in miss.items()}}
        res["readings"] = {"erasure_cause": max(cats, key=cats.get), "missed_region": max(res["missed_repair"], key=res["missed_repair"].get)}
        out[name] = res
        print(json.dumps({name: res}), flush=True)
        del world; torch.cuda.empty_cache()
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
