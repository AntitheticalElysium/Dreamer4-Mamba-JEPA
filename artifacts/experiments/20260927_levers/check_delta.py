"""The thesis's Experiment 0 in the per-tile world (2026-10-04): does Mamba's step size Delta flag the tiles imagination gets
wrong? (Thesis: Delta as a compute-free hallucination gate; the 2026-07 critique: Delta is an input-saliency / overwrite gate,
so it may track where content changes, not where the model errs. Never measured in the per-tile worlds.)
Worlds: the fmamba 36k teacher worlds, seeds 7 and 8 (M6). DEV futures (1,002 roots), self-fed rollout of sample 0's actions
(teval's convention: the 4 true context frames, then windows of up to 5 of its own frames). Per step k and VIEW slot (map tiles
0-62): Delta = softplus(dt + dt_bias) of each layer's Mamba-2 (a forward hook on in_proj; dt = its last nheads outputs), at the
window's LAST time step (the frame just taken in), mean over heads; "mean" averages the 6 layers, "last" is layer 6.
Target: the tile's imagined squared error against the true frame (alive steps), top decile of that world's map-tile errors.
Signals, each as an AUC for the target:
  delta_mean, delta_last       the gate
  input_change                 squared change of the slot's input token from the previous input frame (pure saliency)
  disagreement                 squared difference between the s7 and s8 worlds' imagined tokens (a free 2-member ensemble)
  delta_mean_within_change     AUC of delta_mean inside each input_change quintile, positive-weighted mean
Readings, declared before running:
  delta_flags_error        delta_mean AUC >= 0.70 at both seeds
  delta_beyond_saliency    delta_mean_within_change >= 0.60 at both seeds
  delta_vs_ensemble        delta_mean AUC >= disagreement AUC at both seeds
DELTA_DEVICE=cpu runs it on the CPU with the Mamba-2 reference scan (the module's own float32 equations; added 2026-10-04 while
the GPU is held by E17's Mamba runs).
Usage: check_delta.py <M6 s7.pt> <M6 s8.pt>
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_decision_step as CD  # noqa: E402
T, H = CD.T, CD.H


def auc(s, y):
    # Average ranks for tied scores. Arbitrary argsort ranks make a constant signal look predictive.
    o = s.argsort()
    _, counts = torch.unique_consecutive(s[o], return_counts=True)
    end = counts.cumsum(0)
    tied_ranks = (end.double() + (end - counts).double() + 1) / 2
    r = torch.empty(len(s), dtype=torch.float64)
    r[o] = torch.repeat_interleave(tied_ranks, counts)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def rollout(world, cache, device, config, batch=4):
    """-> imagined frames [R,16,81,192] fp16, Delta [R,16,6,82] (mean over heads), input-change [R,16,81]"""
    grabbed = []
    hooks = [l.mix.core.in_proj.register_forward_hook(lambda m, i, o, l=l: grabbed.append(
        F.softplus(o[:, -1, -l.mix.core.nheads:].float() + l.mix.core.dt_bias.float()).mean(-1))) for l in world.layers]
    R = len(cache["ctx"])
    gen = torch.empty(R, H, 81, 192, dtype=torch.float16); dlt = torch.empty(R, H, len(world.layers), 82)
    chg = torch.empty(R, H, 81)
    with torch.no_grad():
        for i in range(0, R, batch):
            ctx = cache["ctx"][i:i + batch].float().cpu(); b = len(ctx)
            ca, fa = cache["ctx_a"][i:i + batch].cpu(), cache["fut_a"][i:i + batch].cpu()
            g, ah = [ctx[:, j] for j in range(4)], [ca[:, j] for j in range(3)]
            for k in range(H):
                w = min(len(g), 5)
                grabbed.clear()
                nxt = T.step(world, torch.stack(g[-w:], 1), torch.stack(ah[-(w - 1):] + [fa[:, k]], 1), device, config)
                dlt[i:i + b, k] = torch.stack([d.view(b, 82).cpu() for d in grabbed], 1)
                chg[i:i + b, k] = ((g[-1] - g[-2]) ** 2).sum(-1)
                gen[i:i + b, k] = nxt.half()
                g.append(nxt); ah.append(fa[:, k])
    for h in hooks:
        h.remove()
    return gen, dlt, chg


def main():
    from d4mj.config import config_from_dict
    import spatial as Sp
    import dataclasses
    import os
    device = torch.device(os.environ.get("DELTA_DEVICE", "cuda"))
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    cache = T.build_cache("raw", device)
    fut = cache["fut"].float().cpu()                                                          # [R,16,81,192] sample 0
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()                                       # [R,16]
    runs = {}
    for path in sys.argv[1:3]:
        world, st = T.load_world(Path(path), device)
        if device.type == "cpu":
            for l in world.layers:
                l.mix.settings = dataclasses.replace(l.mix.settings, backend="reference")
        runs[st["name"]] = rollout(world, cache, device, config)
        del world; torch.cuda.empty_cache()
    names = list(runs)
    dis = ((runs[names[0]][0].float() - runs[names[1]][0].float()) ** 2).sum(-1)              # [R,16,81]
    out = {}
    for n in names:
        gen, dlt, chg = runs[n]
        err = ((gen.float() - fut) ** 2).sum(-1)                                               # [R,16,81]
        m = alive[:, :, None].expand(-1, -1, 63)
        e = err[:, :, :63][m]
        y = e >= e.quantile(0.9) if len(e) < 2 ** 24 else e >= e[torch.randperm(len(e))[:2 ** 24]].quantile(0.9)
        sig = {"delta_mean": dlt[:, :, :, 1:64].mean(2)[m], "delta_last": dlt[:, :, -1, 1:64][m],
               "input_change": chg[:, :, :63][m], "disagreement": dis[:, :, :63][m]}
        res = {k: auc(v, y) for k, v in sig.items()}
        q = sig["input_change"]
        edges = q.quantile(torch.tensor([0.2, 0.4, 0.6, 0.8]))
        bins = torch.bucketize(q, edges)
        parts = [(auc(sig["delta_mean"][bins == j], y[bins == j]), int(y[bins == j].sum())) for j in range(5)]
        res["delta_mean_within_change"] = sum(a * w for a, w in parts if a == a) / max(sum(w for a, w in parts if a == a), 1)
        res["within_change_by_quintile"] = [round(a, 4) for a, _ in parts]
        res["tiles"] = int(len(y)); res["positives"] = int(y.sum())
        out[n] = res
        print(json.dumps({n: res}), flush=True)
    out["readings"] = {"delta_flags_error": all(out[n]["delta_mean"] >= 0.70 for n in names),
                       "delta_beyond_saliency": all(out[n]["delta_mean_within_change"] >= 0.60 for n in names),
                       "delta_vs_ensemble": all(out[n]["delta_mean"] >= out[n]["disagreement"] for n in names)}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
