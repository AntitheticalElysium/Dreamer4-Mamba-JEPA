"""E14m. Is a deterministic world's imagination "monotone"? (User, 2026-10-02: is a generative / sampling objective (option 3)
needed so that imagined worlds do not look too uniform?) Measured, not assumed: the tile-class content of imagined frames
against the truth, split into content the world could know (cells observable at the root) and content it could not (cells
revealed after the root, on the true camera path).

Data: diagnosis futures (1,002 roots, sample 0 = factual; test-seed roots only, since the probe is fitted on train-seed roots).
Rollout: teval's convention (4 context frames, then a 5-frame window); depths 1, 2, 4, 8, 16 while sample 0 is alive.
Classes: the simulator's tile per map cell (truth); teval's ridge tile probe on the true tokens (probe floor) and on the imagined
tokens. Cells: the 63 map cells minus the player cell, minus cells holding a mob at that depth (simulator), split by the true
camera path (driftanat): observable (world cell inside the root view) / revealed (outside). "aligned": depths where the
imagined camera offset equals the true one (the cells then show the same world locations; accuracy is meaningful there).
Per world, decoding, cell set, variant (all valid depths / aligned only), depth:
  hist         17-class histograms: imagined, true-probe, simulator
  tv           total variation distance imagined vs true-probe; tv_floor = true-probe vs simulator (probe noise)
  entropy      pooled class entropy in bits, imagined vs true-probe
  frame_div    mean number of distinct classes per frame within the set, imagined vs true-probe
  ratio        imagined share / true-probe share, per class with true share >= 0.5%
  majority     the true-probe majority class's share, imagined vs true
  accuracy     imagined class == simulator class (true-probe's own accuracy as its floor); to_majority = P(imagined = majority |
               simulator != majority)
Decodings: every world deterministic (as trained and as teval evaluates); the categorical world also SAMPLED (multinomial over
its K code logits, temperature 1, seed 0): the minimal form of a generative decoding (option 3).
Readings, declared before running (per world and decoding; aligned, depth 16; revealed and observable separately):
  M_monotone           imagined majority share >= true + 0.10, OR imagined entropy <= 0.8 x true entropy
  M_faithful           tv <= max(0.10, 2 x tv_floor) AND imagined entropy >= 0.9 x true entropy
  M_known_drift        M_monotone on OBSERVABLE cells as well (the collapse is then drift of known content, not unseen content)
  M_sampling_restores  (categorical) sampled decoding is M_faithful on revealed cells AND its observable aligned accuracy is no
                       more than 0.05 below argmax's; revealed diversity restored with observable accuracy down > 0.05 =
                       sampling corrupts known content (a generative path would have to be confined to unseen content)
Usage: monotone.py <world.pt> ... -> evals/monotone_<name>.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import driftanat as DA  # noqa: E402
SD, T = DA.SD, DA.T
H = SD.H
DEPTHS = (1, 2, 4, 8, 16)
NAMES = ("invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond", "table",
         "furnace", "sand", "lava", "plant", "ripe_plant")                       # craftax_classic.constants.BlockType


@torch.no_grad()
def rollout(world, ctx, ca, fa, device, config, sample):
    from d4mj.train import autocast_context
    from tworld import quantize
    rng = torch.Generator(device=device).manual_seed(0)
    gen = torch.empty(len(ctx), H, 81, 192, dtype=torch.float16)
    for i in range(0, len(ctx), 16):
        c4, a3, fk = ctx[i:i + 16].float(), ca[i:i + 16], fa[i:i + 16]
        frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
        for k in range(H):
            w = 4 if k == 0 else 5
            fr, ac = torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
            if sample:
                x = world.codes[quantize(fr.to(device), world.codes)]
                with autocast_context(config):
                    p = world(x, ac.to(device))[2][:, -1].float().softmax(-1)                 # [b,81,K]
                g = world.codes[torch.multinomial(p.flatten(0, 1), 1, generator=rng)[:, 0].view(p.shape[:2])].float().cpu()
            else:
                g = T.step(world, fr, ac, device, config)
            gen[i:i + 16, k] = g.half(); frames.append(g); hist.append(fk[:, k])
    return gen


def classes(probes, tok):
    """[R,16,81,192] -> [R,16,63] probe tile classes."""
    flat = tok[:, :, :63].flatten(0, 2)
    return torch.cat([probes.tile(flat[i:i + 65536].float()).argmax(-1) for i in range(0, len(flat), 65536)]).view(*tok.shape[:2], 63)


def summarize(img, tru, simc, m):
    """classes [R,63] at one depth, mask [R,63] -> stats."""
    if not m.any():
        return None
    h = lambda c: torch.bincount(c[m], minlength=17).float()
    hi, ht, hs = h(img), h(tru), h(simc)
    pi, pt, ps = hi / hi.sum(), ht / ht.sum(), hs / hs.sum()
    ent = lambda p: float(-(p[p > 0] * p[p > 0].log2()).sum())
    has = m.any(1)
    div = lambda c: float((F.one_hot(c, 17).bool() & m[..., None]).any(1).sum(-1).float()[has].mean())
    maj = int(pt.argmax())
    other = m & (simc != maj)
    return {"n_cells": int(m.sum()), "n_frames": int(has.sum()),
            "hist_imagined": [round(float(v), 4) for v in pi], "hist_true_probe": [round(float(v), 4) for v in pt],
            "hist_simulator": [round(float(v), 4) for v in ps],
            "tv": 0.5 * float((pi - pt).abs().sum()), "tv_floor": 0.5 * float((pt - ps).abs().sum()),
            "entropy_imagined": ent(pi), "entropy_true": ent(pt), "frame_div_imagined": div(img), "frame_div_true": div(tru),
            "majority_class": NAMES[maj], "majority_imagined": float(pi[maj]), "majority_true": float(pt[maj]),
            "ratio": {NAMES[c]: round(float(pi[c] / pt[c]), 3) for c in range(17) if pt[c] >= 0.005},
            "accuracy": float((img[m] == simc[m]).float().mean()), "accuracy_true_probe": float((tru[m] == simc[m]).float().mean()),
            "to_majority": float((img[other] == maj).float().mean()) if other.any() else None}


def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache("raw", torch.device("cpu"))
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    ctx, ca, fa, fut = cache["ctx"], cache["ctx_a"], cache["fut_a"], cache["fut"]
    R, root = len(ctx), ctx[:, -1]
    test = ~train_roots
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()                                  # [R,16] sample 0 alive
    prev = torch.cat([root[:, None], fut[:, :-1]], 1)
    true_off = DA.offsets(torch.cat([estimate(prev[i:i + 32].float(), fut[i:i + 32].float()) for i in range(0, R, 32)]))
    vis = meta["future_visible"][:, 0].float()
    simc = T.facts_of(vis)["tile"]                                                        # [R,16,63]
    mob = vis[..., 1071:1512].reshape(R, H, 7, 9, 7).sum(-1).flatten(2) > 0
    tru = classes(probes, fut)
    rr = torch.arange(7)[:, None].expand(7, 9).reshape(63); cc = torch.arange(9)[None].expand(7, 9).reshape(63)
    wr, wc = rr + true_off[..., 0:1], cc + true_off[..., 1:2]
    observable = (wr >= 0) & (wr < 7) & (wc >= 0) & (wc < 9)                             # [R,16,63]
    base = test[:, None, None] & alive[..., None] & ~mob
    base[..., 31] = False
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        res = {"world": name, "decodings": {}}
        for decoding in (("argmax", "sampled") if world.head == "categorical" else ("deterministic",)):
            gen = rollout(world, ctx, ca, fa, device, config, decoding == "sampled")
            gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
            img_off = DA.offsets(torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)]))
            aligned = (img_off == true_off).all(-1)                                       # [R,16]
            img = classes(probes, gen)
            d = {"aligned_share": {k: float(aligned[:, k - 1][test & alive[:, k - 1]].float().mean()) for k in DEPTHS}}
            for cell_set, cm in (("observable", observable), ("revealed", ~observable)):
                for variant, vm in (("all", torch.ones_like(aligned)), ("aligned", aligned)):
                    m = base & cm & vm[..., None]
                    d[f"{cell_set}_{variant}"] = {k: summarize(img[:, k - 1], tru[:, k - 1], simc[:, k - 1], m[:, k - 1]) for k in DEPTHS}
            res["decodings"][decoding] = d
            print(json.dumps({"world": name, "decoding": decoding, "aligned_share": d["aligned_share"],
                              "d16": {s: {q: d[s][16][q] for q in ("n_cells", "tv", "tv_floor", "entropy_imagined", "entropy_true",
                                                                   "majority_class", "majority_imagined", "majority_true", "accuracy")}
                                      for s in ("observable_aligned", "revealed_aligned") if d[s][16]}}), flush=True)
            del gen
        rd = {}
        for decoding, d in res["decodings"].items():
            for cell_set in ("observable", "revealed"):
                s = d[f"{cell_set}_aligned"][16]
                if s is None:
                    continue
                rd[f"{decoding}_{cell_set}_monotone"] = s["majority_imagined"] >= s["majority_true"] + 0.10 or s["entropy_imagined"] <= 0.8 * s["entropy_true"]
                rd[f"{decoding}_{cell_set}_faithful"] = s["tv"] <= max(0.10, 2 * s["tv_floor"]) and s["entropy_imagined"] >= 0.9 * s["entropy_true"]
            rd[f"{decoding}_known_drift"] = rd.get(f"{decoding}_observable_monotone", False) and rd.get(f"{decoding}_revealed_monotone", False)
        if "sampled" in res["decodings"]:
            acc = lambda dec: res["decodings"][dec]["observable_aligned"][16]["accuracy"]
            rd["sampling_restores"] = rd.get("sampled_revealed_faithful", False) and acc("sampled") >= acc("argmax") - 0.05
            rd["sampling_corrupts_known"] = rd.get("sampled_revealed_faithful", False) and acc("sampled") < acc("argmax") - 0.05
        res["readings"] = rd
        (HERE / "evals" / f"monotone_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, "readings": rd}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
