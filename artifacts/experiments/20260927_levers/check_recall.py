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
v3 --futures (2026-10-04, declared before running): the diagnosis futures' TRUE trajectories (DEV roots, 4 context frames + each
of the 5 sampled 16-step futures under the same actions: 5,010 x 20 frames; targets whose frame is past a death excluded). Each
target t is predicted from the last W frames (sliding, teval's convention), W = the world's trained context (time rows - 1: 5
for the 6-frame worlds, 15 for E17's L = 16) or --window W. Groups by the TRUE sighting age and slot: same_2_5, moved_2_5,
same_6_15, moved_6_15 (ages beyond a world's window are scored too: what it does without the sighting).
  futures_replicates   (6-frame parents) Mamba's same_2_5 capture - attention's >= 0.10 at s7 36k and at s8 30k
  E17 stage 2 (amendment 2, NOTEBOOK), per seed, window 15:
  long_recall_same     M16's same_6_15 capture >= 0.5 and >= A16's + 0.10 at both seeds
  long_recall_used     M16's same_6_15 capture at window 15 - at window 5 >= 0.2 at both seeds
  moved_unsolved       moved_6_15 capture <= 0.35 for every world (descriptive)
v4 --futures --imagined (declared before running): the same groups in the world's SELF-FED rollout (teval's convention: the 4
true context frames, then windows of up to W of its own frames under the true actions; sample 0's trajectory, 1,002 roots),
scored only at steps whose imagined camera offset still equals the true one (cumulative scroll.estimate shifts), so imagined
and true cells coincide; sightings may lie in imagined frames (the world must keep its own imagination consistent).
  imagined_recall_edge   (6-frame parents) Mamba's same_2_5 capture - attention's >= 0.10 at s7 36k and at s8 30k
Run v4 (lane71): imagined_recall_edge TRUE (same_2_5 Mamba - attention: s7 36k +0.18, s8 30k +0.21, s8 36k +0.32), but the
no-memory baselines also differ in imagination (same_6_15: attention -0.14 / -0.25 / -0.19, Mamba -0.06 / -0.08 / -0.01).
v5 memory contrast (declared before running): the SAME cells, the same world, teacher-forced, with the sighting visible
(window 5) and not (--window 1: the current frame only, in-distribution: teacher training predicts from one frame):
  recall_gain_same = capture(same_2_5, w5) - capture(same_2_5, w1)
  memory_contrast    Mamba's recall_gain_same - attention's >= 0.10 at s7 36k and at s8 30k (computed from lane70's w5 lines
                     and this run's w1 lines, both in path order)
Usage: check_recall.py [--futures [--imagined]] [--window W] <world.pt> ...
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


def futures(device):
    """[R*5, 20, 81, 192] true trajectories, actions [R*5, 19], valid targets [R*5, 20] (frame alive)"""
    import check_decision_step as CD
    fut5, _ = CD.SD.token_cache(device)
    cache = T.build_cache("raw", device)
    meta, _, _ = T.split()
    s = torch.cat([cache["ctx"].cpu()[:, None].expand(-1, 5, -1, -1, -1), fut5.cpu()], 2).flatten(0, 1)
    a = torch.cat([cache["ctx_a"].cpu(), cache["fut_a"].cpu()], 1).repeat_interleave(5, 0)
    alive = ~meta["future_dead"].cumsum(2).bool()                                                 # [R,5,16]
    return s, a, torch.cat([torch.ones(alive.shape[0] * 5, 4, dtype=torch.bool), alive.flatten(0, 1)], 1)


def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    argv = sys.argv[1:]
    fut, imagined = "--futures" in argv, "--imagined" in argv
    window = int(argv[argv.index("--window") + 1]) if "--window" in argv else None
    paths = [x for i, x in enumerate(argv) if x.endswith(".pt")]
    if fut:
        seqs, acts, valid = futures(device)
        if imagined:
            seqs, acts, valid = seqs[0::5], acts[0::5], valid[0::5]                              # sample 0: the factual future
    else:
        pool = torch.load(TW.POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
        main_rows = torch.where(~pool["terminal"])[0]
        held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
        batches = [held[i:i + 32] for i in range(0, len(held), 32)]
    out = {}
    for path in paths:
        world, st = T.load_world(Path(path), device)
        name = st["name"] + (f"_w{window}" if window else "") + ("_imagined" if imagined else "")
        W = window or world.time.shape[0] - 1
        if fut:                                                    # per-token SSMs: smaller batches (memory; same per-sequence math)
            bs = 16 if getattr(world, "backbone_kind", "full") in ("fmamba", "fcanvas") else 64
            batches = [(seqs[i:i + bs], acts[i:i + bs], valid[i:i + bs]) for i in range(0, len(seqs), bs)]
        acc = {c: {"n": 0, "world": 0.0, "sighting": 0.0, "neighbour": 0.0}
               for c in ("recallable", "unseen", "same_slot", "moved_slot", "age2", "age3plus", "same_2_5", "moved_2_5", "same_6_15", "moved_6_15")}
        with torch.no_grad():
            for batch in batches:
                if imagined:
                    s, a, ok = batch
                    g = [s[:, j].float() for j in range(4)]
                    for t in range(4, s.shape[1]):
                        w = min(len(g), W)
                        g.append(T.step(world, torch.stack(g[-w:], 1), a[:, t - w:t], device, config))
                    gen, s = torch.stack(g, 1), s.float()
                    pred = gen[:, 1:]                                                            # pred[:, t-1] = frame t
                    io = torch.tensor(SHIFTS)[estimate(gen[:, :-1], gen[:, 1:])].cumsum(1)         # offset of frame t, t >= 1
                    to = torch.tensor(SHIFTS)[estimate(s[:, :-1], s[:, 1:])].cumsum(1)
                    ok = ok & torch.cat([torch.zeros(len(s), 4, dtype=torch.bool), (io == to).all(-1)[:, 3:]], 1)
                elif fut:
                    s, a, ok = batch
                    pred = torch.stack([T.step(world, s[:, max(0, t - W):t].float(), a[:, max(0, t - W):t], device, config)
                                        for t in range(1, s.shape[1])], 1)                       # pred[:, t-1] = frame t
                    s = s.float()
                else:
                    b = Sp.batch_of(pool, batch, "tokens", device)
                    s, a = b["s"].float(), b["actions"]
                    with autocast_context(config):
                        pred = world(s, F.pad(a, (0, 1)))[0].float()                             # pred[:, t] = frame t+1
                    s, pred = s.cpu(), pred.cpu()
                    ok = torch.ones(s.shape[:2], dtype=torch.bool)
                for j in range(len(s)):
                    t, ce, cls, sf, sc, nb = cells(s[j])
                    live = ok[j, t]
                    tgt = s[j, t, ce]
                    errs = {"world": (pred[j, t - 1, ce] - tgt).square().sum(-1), "sighting": (s[j, sf, sc] - tgt).square().sum(-1),
                            "neighbour": (s[j, t, nb] - tgt).square().sum(-1)}
                    rec, age, same = (cls == 1) & live, t - sf, sc == ce
                    groups = {"recallable": rec, "unseen": (cls == 0) & live, "same_slot": rec & same, "moved_slot": rec & ~same,
                              "age2": rec & (age == 2), "age3plus": rec & (age > 2), "same_2_5": rec & same & (age <= 5),
                              "moved_2_5": rec & ~same & (age <= 5), "same_6_15": rec & same & (age >= 6) & (age <= 15),
                              "moved_6_15": rec & ~same & (age >= 6) & (age <= 15)}
                    for g, m in groups.items():                    # (v3-v4 reused `name` here: lane70's lines are keyed "moved_6_15", in path order)
                        acc[g]["n"] += int(m.sum())
                        for e, v in errs.items():
                            acc[g][e] += float(v[m].sum())
        res = {c: {"n": v["n"], **{e: v[e] / max(v["n"], 1) for e in ("world", "sighting", "neighbour")}} for c, v in acc.items()}
        for g in ("recallable", "same_slot", "moved_slot", "age2", "age3plus", "same_2_5", "moved_2_5", "same_6_15", "moved_6_15"):
            r = res[g]
            r["recall_capture"] = (r["neighbour"] - r["world"]) / (r["neighbour"] - r["sighting"]) if r["n"] else None
        res["unseen"]["gain_over_neighbour"] = 1 - res["unseen"]["world"] / res["unseen"]["neighbour"]
        out[name] = res
        print(json.dumps({name: res}), flush=True)
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
    if fut:
        rd = {}
        pairs = (("corrt_raw_teacher_s7_u36000", "corrt_raw_teacher_s7_fmamba_u36000"),
                 ("corrt_raw_teacher_s8_u36000_at30000", "corrt_raw_teacher_s8_fmamba_u36000_at30000"))
        x = "_imagined" if imagined else ""
        if all(a + x in out and m + x in out for a, m in pairs):
            key = "imagined_recall_edge" if imagined else "futures_replicates"
            rd[key] = all(out[m + x]["same_2_5"]["recall_capture"] - out[a + x]["same_2_5"]["recall_capture"] >= 0.10 for a, m in pairs)
    out["readings"] = rd
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
