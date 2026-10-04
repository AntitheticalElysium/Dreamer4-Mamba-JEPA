"""Item 9 follow-up (2026-10-02): how often does a world draw a faced obstacle REMOVED at a DO step when it truly stays, one step
from true inputs vs self-fed? (check_erasure: in DO-learning worlds 73-86% of the erasures before false scrolls happen at a DO on
the faced cell; consfit's one-step hallucination on held pool windows is only 0.7-1.5%.)
Diagnosis futures, sample 0, simulator truth. DO steps (action 5) whose faced cell (simulator facing at the current frame) holds a
blocking class that is UNCHANGED in the next frame ("no-change DO on an obstacle"), and DO steps whose faced cell truly changes
("consequence DO"). The faced cell's class in the world's predicted next frame (teval tile probe):
  teacher   predicted from the TRUE window (4 context frames + true future frames, teval's window)
  selffed   predicted inside the world's own rollout (teval convention), at steps where the imagined view is still aligned
Reported: removal rate (drawn passable while it truly stays) per mode, overall and by true class; catch rate on consequence DOs.
Readings, declared before running (per world):
  selffed_amplifies   selffed removal rate >= 2 x the teacher-forced one
  learned_do_halluc   (per seed) the 36k world's teacher-forced removal rate >= 3 x the 18k world's
Usage: check_dohalluc.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_decision_step as CD
DA, SD, T = CD.DA, CD.SD, CD.T
H, SOLID = CD.H, CD.SOLID
FACED = {0: 30, 1: 32, 2: 22, 3: 40}
NAMES = {2: "grass", 3: "water", 4: "stone", 5: "tree", 8: "coal", 9: "iron", 10: "diamond", 11: "table", 12: "furnace", 15: "plant", 16: "ripe_plant", 1: "oob"}


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
    prev = torch.cat([root[:, None], fut5[:, 0, :-1]], 1)
    ts0 = torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, 0].float()) for i in range(0, R, 16)])
    true_off = DA.offsets(ts0).long()                                                     # [R,16,2]
    allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()
    tiles = allv[:, :, :1071].reshape(R, 17, 63, 17).argmax(-1)
    facing = allv[:, :, 1516:1520].argmax(-1)
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    fut0 = fut5[:, 0]
    # DO steps k (frame k -> k+1; frame index 0 = root, k >= 1 = fut0[k-1])
    sel = []
    for r in range(R):
        for k in range(H):
            if int(fa[r, k]) != 5 or not bool(alive[r, k]):
                continue
            cell = FACED[int(facing[r, k])]
            before, after = int(tiles[r, k, cell]), int(tiles[r, k + 1, cell])
            sel.append((r, k, cell, before, after))
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
        img_off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])).long()
        aligned = (img_off == true_off).all(-1)                                            # after step k
        stats = {"teacher": {}, "selffed": {}}
        for j in range(0, len(sel), 128):
            chunk = sel[j:j + 128]
            wins, acts = [], []
            for r, k, cell, before, after in chunk:
                frames_true = torch.cat([ctx[r].float(), fut0[r, :k].float()], 0)
                w = 4 if k == 0 else 5
                wins.append(frames_true[-w:]); acts.append(torch.cat([ca[r], fa[r, :k + 1]], 0)[-w:])
            for w in (4, 5):
                idx = [q for q in range(len(chunk)) if wins[q].shape[0] == w]
                if not idx:
                    continue
                pt = T.step(world, torch.stack([wins[q] for q in idx]), torch.stack([acts[q] for q in idx]), device, config)
                cls_t = probes.tile(torch.stack([pt[n, chunk[q][2]] for n, q in enumerate(idx)])).argmax(-1)
                for n, q in enumerate(idx):
                    r, k, cell, before, after = chunk[q]
                    ok_self = k == 0 or bool(aligned[r, k - 1])
                    cls_s = int(probes.tile(gen[r, k, cell][None].float()).argmax(-1)) if ok_self else None
                    kind = "stay" if (before == after and before in SOLID) else ("change" if before != after else "other")
                    for mode, c in (("teacher", int(cls_t[n])), ("selffed", cls_s)):
                        if c is None or kind == "other":
                            continue
                        key = (kind, NAMES.get(before, str(before)))
                        s = stats[mode].setdefault(key, [0, 0])
                        s[0] += 1
                        s[1] += int(c not in SOLID) if kind == "stay" else int(c == after)
        res = {}
        for mode in stats:
            st_ = stats[mode]
            stay = [v for (kd, _), v in st_.items() if kd == "stay"]; chg = [v for (kd, _), v in st_.items() if kd == "change"]
            res[mode] = {"stay_n": sum(v[0] for v in stay), "removal_rate": round(sum(v[1] for v in stay) / max(sum(v[0] for v in stay), 1), 4),
                         "change_n": sum(v[0] for v in chg), "caught_rate": round(sum(v[1] for v in chg) / max(sum(v[0] for v in chg), 1), 4),
                         "by_class": {f"{kd}:{c}": {"n": v[0], "rate": round(v[1] / max(v[0], 1), 4)} for (kd, c), v in sorted(st_.items())}}
        res["readings"] = {"selffed_amplifies": res["selffed"]["removal_rate"] >= 2 * max(res["teacher"]["removal_rate"], 1e-9)}
        out[name] = res
        print(json.dumps({name: res}), flush=True)
        del world; torch.cuda.empty_cache()
    for s in (7, 8):
        a, b = out.get(f"corrt_raw_teacher_s{s}_u18000"), out.get(f"corrt_raw_teacher_s{s}_u36000")
        if a and b:
            out[f"learned_do_halluc_s{s}"] = b["teacher"]["removal_rate"] >= 3 * max(a["teacher"]["removal_rate"], 1e-9)
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
