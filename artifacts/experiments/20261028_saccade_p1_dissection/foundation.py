"""Exact real-path cache plus paired TSP/fovea reconstruction measurements. Frozen world; no reader fitting."""
import time

import numpy as np
import torch

from .common import *


@torch.no_grad()
def encode(model, L, cfg):
    result = []
    for e, ep in enumerate(s1.episodes(L)):
        rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
        recs = [[(0, 1, t, *D.fixation_prior(rng, ())) for t in range(ep.T + 1)]]
        c = TR.tensors(D.gather([ep], recs), "cuda")
        with TR.autocast(torch.device("cuda"), cfg):
            ps = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            P = M.Percept(*(torch.cat(z, 1) for z in zip(*[p.parts() for p in ps])), c["fixation"])
            emb = model.belief.percept_embed(P)
            x = model.belief.tokens(emb, c["reward"], c["terminated"], c["action"])
            y, _ = model.belief(x, model.belief.initial(1, "cuda"))
            b, bp = y[:, 0::2], torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            cells = M.window(c["fixation"])
            pixels = M.cell_pixels(M.crop(c["frames"], c["fixation"]))
            mt, st = model.pixels(model.tsp.predict(bp, P.p, P.v, cells), P.v, cells)
            mr, sr = model.pixels(model.tsp.reconstruct(P.g, P.v, cells), P.v, cells)
            assert torch.equal(st, sr)
            aux = {"tsp_nll": M.pixel_nll(mt, st, pixels).mean(-1), "rec_nll": M.pixel_nll(mr, sr, pixels).mean(-1),
                   "tsp_mae": (((mt.float() + 1) * 127.5).clamp(0, 255) - pixels.float()).abs().mean(-1),
                   "rec_mae": (((mr.float() + 1) * 127.5).clamp(0, 255) - pixels.float()).abs().mean(-1),
                   "sigma_rgb": (st.float().clamp(min=-7).exp() * 127.5).mean(-1)}
        row = {"p": P.p[0].float().cpu(), "g": P.g[0].float().cpu(), "v": P.v[0].float().cpu(),
               "b": b[0].float().half().cpu(), "bp": bp[0].float().half().cpu(), "x": x[0, 0::2].float().half().cpu(),
               **{k: c[k][0].cpu() for k in ("fixation", "prev_action", "action", "reward", "terminated")},
               **{k: v[0].float().cpu() for k, v in aux.items()}}
        row["fix"] = row.pop("fixation")
        result.append(row)
        if e % 100 == 0:
            print(f"real cache {e}/{len(L['ep_T'])}", flush=True)
    return result


if __name__ == "__main__":
    before = fingerprints(); save("frozen_identity.json", before)
    model, cfg = load_model(); h = model_hash(model)
    start = time.time()
    for split in ("train", "dev"):
        L = labels(split)
        enc = encode(model, L, cfg)
        torch.save(enc, OUT / f"real_{split}.pt")
        # Reproduction checks against the unchanged S1 encoder on two real episodes.
        ref = s1.encode(model, s1.first_episodes(L, 2), cfg)
        diffs = {k: max(float((enc[i][k].half() - ref[i][k]).abs().max()) for i in range(2))
                 for k in ("p", "g", "v", "b", "bp")}
        assert max(diffs.values()) == 0, diffs
        save(f"cache_{split}_identity.json", {"episodes": len(enc), "states": sum(len(r["p"]) for r in enc),
             "s1_half_cache_max_differences": diffs, "shared_decoder_scales_identical": True})
        del enc, L, ref
    assert model_hash(model) == h and fingerprints() == before
    save("foundation_run.json", {"seconds": time.time() - start, "frozen_parameter_hash": h,
         "checkpoint_and_sources_unchanged": True, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30})
