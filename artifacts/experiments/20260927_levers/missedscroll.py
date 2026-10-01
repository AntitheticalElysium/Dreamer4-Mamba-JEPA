"""E11c. Why does the move decision fail in imagination? (E11b: 86% of first view-position errors are missed scrolls on moves that
succeeded; they almost never happen from true frames, 0.3% at depth 1, then ~3-4% of remaining roots per step.)

Same rollouts as driftanat.py (diagnosis futures, imagined rollout of the shared actions, teval's window convention, scroll
offsets from scroll.estimate). Cases, on depths where all five samples are alive and agree on the true offset:
  missed    the first step at which the imagined offset goes wrong, where the factual action is a move that succeeded
            (true scroll != none) and the imagined frame did not scroll
  control   moves that succeeded and that the imagined frame DID scroll correctly, at steps where the offset was still right,
            matched to the missed cases' depth distribution (up to 3 per missed case, seeded)
For each case the head's move logit (blockwin.py: frame logit + target-tile gate for corrt; frame logit for corrg) is read on:
  imagined   the window the rollout used (imagined frames after the root)
  true       the same positions filled with sample 0's TRUE frames
  img_hist   true current frame, imagined history
  img_cur    imagined current frame, true history
A positive logit = scroll. Also, at the target tile (the cell the move enters) of the current frame: squared distance between
the imagined and true tokens, beside the same distance for the player tile and for the frame mean.
Readings, declared before running:
  input_caused     on missed cases, true > 0 for >= 80% while imagined <= 0 (the same weights decide right on true input)
  current_frame    img_cur reproduces the miss (<= 0) on >= 70% of missed cases while img_hist does not (> 0 on >= 70%)
  history          the reverse
  target_tile      the target tile's imagined-vs-true distance on missed cases >= 2x that on control cases
Usage: missedscroll.py <world.pt> ... -> evals/missedscroll_<name>.json
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
TARGET = {1: 30, 2: 32, 3: 22, 4: 40}                       # left, right, up, down: the tile the move enters


@torch.no_grad()
def move_logit(world, frames, acts, targets, device, config):
    from d4mj.train import autocast_context
    with autocast_context(config):
        h, ha = world.backbone_full(frames.to(device), acts.to(device))
        logit = world.frame(ha[:, -1]).float()[:, 0]
        if hasattr(world, "target_gate"):
            logit = logit + world.target_gate(h[torch.arange(len(frames), device=device), -1, targets.to(device)]).float()[:, 0]
    return logit.cpu()


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    true_shift = []
    for s in range(S):
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        true_shift.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(true_shift, 1)
    true_off = DA.offsets(true_shift)
    agree = (true_off == true_off[:, :1]).all(-1).all(1)
    valid = alive & agree
    fut0 = fut5[:, 0]
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        if not hasattr(world, "frame"):
            print(json.dumps({"world": name, "skipped": "no frame-level move logit"}), flush=True)
            continue
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
        img_off = DA.offsets(img_shift)
        aligned = (img_off == true_off[:, 0]).all(-1)
        prev_ok = torch.cat([torch.ones(R, 1, dtype=torch.bool), aligned[:, :-1]], 1)   # offset right before this step
        moved_ok = (true_shift[:, 0] != 0) & (fa >= 1) & (fa <= 4)
        missed = valid & prev_ok & ~aligned & moved_ok & (img_shift == 0)
        first = missed & (torch.cumsum(missed.int(), 1) == 1) & (torch.cumsum((valid & ~aligned).int(), 1) == 1)
        first[:, 0] = False                       # depth 1 uses a 4-frame window; padding would move time positions
        control_pool = valid & prev_ok & aligned & moved_ok & (img_shift == true_shift[:, 0])
        miss_idx = torch.nonzero(first)                                                # [n,2] (root, depth)
        gen_rng = torch.Generator().manual_seed(20261001)
        ctrl = []
        for k in miss_idx[:, 1].unique().tolist():
            n_k = int((miss_idx[:, 1] == k).sum())
            pool = torch.nonzero(control_pool[:, k])[:, 0]
            pick = pool[torch.randperm(len(pool), generator=gen_rng)[:3 * n_k]]
            ctrl += [(int(r), k) for r in pick]
        cases = {"missed": [(int(r), int(k)) for r, k in miss_idx], "control": ctrl}
        res = {"world": name, "n": {c: len(v) for c, v in cases.items()}}
        for cname, lst in cases.items():
            if not lst:
                continue
            variants = {v: [] for v in ("imagined", "true", "img_hist", "img_cur")}
            dist = {"target": [], "player": [], "frame": []}
            for j in range(0, len(lst), 64):
                chunk = lst[j:j + 64]
                wi, wt, aw, tg = [], [], [], []
                for r, k in chunk:
                    full_i = torch.cat([ctx[r].float(), gen[r, :k].float()], 0)       # frames up to the current (k-1)
                    full_t = torch.cat([ctx[r].float(), fut0[r, :k].float()], 0)
                    acts = torch.cat([ca[r], fa[r, :k + 1]], 0)                         # outgoing actions of those frames
                    wi.append(full_i[-5:]); wt.append(full_t[-5:]); aw.append(acts[-5:])
                    tg.append(TARGET[int(fa[r, k])])
                wi, wt, aw, tg = torch.stack(wi), torch.stack(wt), torch.stack(aw), torch.tensor(tg)
                hyb_hist = wi.clone(); hyb_hist[:, -1] = wt[:, -1]                      # imagined history, true current
                hyb_cur = wt.clone(); hyb_cur[:, -1] = wi[:, -1]                        # true history, imagined current
                for v, fr in (("imagined", wi), ("true", wt), ("img_hist", hyb_hist), ("img_cur", hyb_cur)):
                    variants[v].append(move_logit(world, fr, aw, tg, device, config))
                d = ((wi[:, -1] - wt[:, -1]) ** 2).sum(-1)                            # [b,81]
                dist["target"].append(d[torch.arange(len(chunk)), tg]); dist["player"].append(d[:, 31]); dist["frame"].append(d[:, :63].mean(1))
            logits = {v: torch.cat(x) for v, x in variants.items()}
            dd = {q: torch.cat(x) for q, x in dist.items()}
            res[cname] = {"scroll_rate": {v: float((x > 0).float().mean()) for v, x in logits.items()},
                          "mean_logit": {v: float(x.mean()) for v, x in logits.items()},
                          "token_distance_imagined_vs_true": {q: float(x.mean()) for q, x in dd.items()},
                          "depth_hist": torch.bincount(torch.tensor([k for _, k in lst]), minlength=H).tolist()}
        m = res.get("missed")
        if m:
            sr = m["scroll_rate"]
            res["readings"] = {
                "input_caused": sr["true"] >= 0.8 and sr["imagined"] <= 0.2,
                "current_frame": (1 - sr["img_cur"]) >= 0.7 and sr["img_hist"] >= 0.7,
                "history": (1 - sr["img_hist"]) >= 0.7 and sr["img_cur"] >= 0.7,
                "target_tile": ("control" in res and m["token_distance_imagined_vs_true"]["target"] >=
                                2 * res["control"]["token_distance_imagined_vs_true"]["target"])}
        (out_dir / f"missedscroll_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "n": res["n"], "missed": m and m["scroll_rate"], "readings": res.get("readings")}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
