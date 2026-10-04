"""Can each backbone recall terrain across a scroll? (2026-10-04, Mamba diagnosis.) check_memory: the only map content a longer
history supplies is terrain re-entering the view (11.2% of entering cells were seen within 5 frames, 22.2% within 15; a perfect
memory removes 2.1% / 3.9% of a copy world's error). fmamba scans each of the 82 VIEW slots over time: after a scroll a slot
shows another world cell, so its state is misaligned with what re-enters; the full attention backbone attends to every token
of every frame; fcanvas scans world-aligned canvas cells (scroll.estimate's offsets).
Held-out main windows of the raw pool (the 2,048 rows every TWorld run holds out: seed-1 permutation), teacher-forced full
6-frame pass (prediction of frame t+1 from frames 0..t, as trained). On every true scroll (scroll.estimate on the true pair),
the entering cells of frame t+1 are split into
  recallable   the world cell was in view in one of frames 0..t-1 (check_memory's coordinates; most recent sighting)
  unseen       never in view in the window
Per world and class: squared error of the prediction; scale: copying the most recent sighting (perfect memory) and the inward
neighbour (memoryless copy); recall_capture = (neighbour - world) / (neighbour - sighting) on recallable cells.
Readings, declared before running:
  recall_gap_36k     fmamba 36k's recallable error >= 1.10 x the full 36k world's while their unseen errors differ by < 5%
  canvas_recalls_6k  fcanvas 6k's recallable error <= 0.90 x fmamba 6k's (suffix arms, seed 7, lane 9)
Run 1 (lane67) and the curves (lane68, check_recall_curve.log): both readings FALSE in the other direction. fmamba recalls
FASTER: capture s7 0.78 vs 0.47 at 36k, s8 0.61 vs 0.30 at 30k (attention s8 stays 0.30 to 100k; s7 attention reaches 0.74 at
50k, 0.83 at 100k); unseen cells tie; all backbones ~0.18-0.21 at 6k.
v2 split (added after the curves, declared before running it): a cell that left through an edge and re-enters through the same
edge with no perpendicular move re-enters its old VIEW slot, where a per-slot scan is aligned.
  same_slot / moved_slot   recallable cells whose last sighting was at the same view slot / another one
  age2 / age3plus          sighting 2 frames / 3-5 frames before the target
  slot_bias   (per fmamba vs full pair) fmamba's capture edge on same_slot cells >= 2 x its edge on moved_slot cells
Usage: check_recall.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_memory as CM  # noqa: E402
import teval as T  # noqa: E402
import tworld as TW  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402


def cells(x):
    """one window x [6,81,192] -> (target frame t+1, cell, class 1 recallable / 0 unseen, sighting frame, sighting cell)"""
    sh = estimate(x[:-1], x[1:])
    off = torch.cat([torch.zeros(1, 2, dtype=torch.long), torch.tensor(SHIFTS)[sh].cumsum(0)])
    _, enter = CM.source_index()
    te, ce = torch.where(enter[sh])
    t = te + 1
    wr, wc = ce // 9 + off[t, 0], ce % 9 + off[t, 1]
    lr, lc = wr[:, None] - off[None, :, 0], wc[:, None] - off[None, :, 1]
    frames = torch.arange(len(x))
    inview = (lr >= 0) & (lr < 7) & (lc >= 0) & (lc < 9) & (frames[None] < t[:, None])
    s = torch.where(inview, frames[None], -1).max(1).values
    sc = lr.gather(1, s.clamp(min=0)[:, None])[:, 0].clamp(0, 6) * 9 + lc.gather(1, s.clamp(min=0)[:, None])[:, 0].clamp(0, 8)
    dr, dc = torch.tensor(SHIFTS)[sh[te]].unbind(-1)
    nb = (ce // 9 - dr).clamp(0, 6) * 9 + (ce % 9 - dc).clamp(0, 8)                             # inward neighbour, frame t+1
    return t, ce, (s >= 0).long(), s.clamp(min=0), sc, nb


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    pool = torch.load(TW.POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    out = {}
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        acc = {c: {"n": 0, "world": 0.0, "sighting": 0.0, "neighbour": 0.0}
               for c in ("recallable", "unseen", "same_slot", "moved_slot", "age2", "age3plus")}
        with torch.no_grad():
            for i in range(0, len(held), 32):
                b = Sp.batch_of(pool, held[i:i + 32], "tokens", device)
                s, a = b["s"].float(), b["actions"]
                with autocast_context(config):
                    pred = world(s, F.pad(a, (0, 1)))[0].float()                                 # pred[:, t] = frame t+1
                s, pred = s.cpu(), pred.cpu()
                for j in range(len(s)):
                    t, ce, cls, sf, sc, nb = cells(s[j])
                    tgt = s[j, t, ce]
                    errs = {"world": (pred[j, t - 1, ce] - tgt).square().sum(-1), "sighting": (s[j, sf, sc] - tgt).square().sum(-1),
                            "neighbour": (s[j, t, nb] - tgt).square().sum(-1)}
                    rec = cls == 1
                    groups = {"recallable": rec, "unseen": ~rec, "same_slot": rec & (sc == ce), "moved_slot": rec & (sc != ce),
                              "age2": rec & (t - sf == 2), "age3plus": rec & (t - sf > 2)}
                    for name, m in groups.items():
                        acc[name]["n"] += int(m.sum())
                        for e, v in errs.items():
                            acc[name][e] += float(v[m].sum())
        res = {c: {"n": v["n"], **{e: v[e] / max(v["n"], 1) for e in ("world", "sighting", "neighbour")}} for c, v in acc.items()}
        for g in ("recallable", "same_slot", "moved_slot", "age2", "age3plus"):
            r = res[g]
            r["recall_capture"] = (r["neighbour"] - r["world"]) / (r["neighbour"] - r["sighting"]) if r["n"] else None
        res["unseen"]["gain_over_neighbour"] = 1 - res["unseen"]["world"] / res["unseen"]["neighbour"]
        out[st["name"]] = res
        print(json.dumps({st["name"]: res}), flush=True)
        del world; torch.cuda.empty_cache()
    rd = {}
    full, fm = out.get("corrt_raw_teacher_s7_u36000"), out.get("corrt_raw_teacher_s7_fmamba_u36000")
    if full and fm:
        rd["recall_gap_36k"] = (fm["recallable"]["world"] >= 1.10 * full["recallable"]["world"]
                                and abs(fm["unseen"]["world"] / full["unseen"]["world"] - 1) < 0.05)
    cv, f6 = out.get("corrt_raw_suffix_s7_fcanvas"), out.get("corrt_raw_suffix_s7_fmamba")
    if cv and f6:
        rd["canvas_recalls_6k"] = cv["recallable"]["world"] <= 0.90 * f6["recallable"]["world"]
    for a, m in (("corrt_raw_teacher_s7_u36000", "corrt_raw_teacher_s7_fmamba_u36000"),
                 ("corrt_raw_teacher_s8_u36000_at30000", "corrt_raw_teacher_s8_fmamba_u36000_at30000")):
        if a in out and m in out:
            edge = {g: out[m][g]["recall_capture"] - out[a][g]["recall_capture"] for g in ("same_slot", "moved_slot")}
            rd[f"slot_bias_{m}"] = edge["same_slot"] >= 2 * edge["moved_slot"]
    out["readings"] = rd
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
