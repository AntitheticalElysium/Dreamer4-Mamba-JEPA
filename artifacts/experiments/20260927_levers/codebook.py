"""E2. Codebooks over layer-normed patch tokens, and what quantizing TRUE tokens costs before any world is trained.

k-means (Lloyd, 30 iterations, init = K random tokens, seed 0) on 500,000 tokens sampled from the training windows
of the chosen pool (the 2,048 held-out main windows excluded, as tworld.py). K in {1024, 4096}.
Quality on the diagnostic futures' TRUE tokens (teval cache), test seeds:
  rel_error    ||q(x) - x||^2 / ||x - mean||^2
  facts        teval.Probes read on q(x) vs on x (tile near / interior / edge, zombie AUC, health, facing)
  flip_rate    among consecutive factual frames whose 63 map tiles and mobs are ALL unchanged (no scroll, no
               edit), the fraction of map cells whose code changes -- codes over contextual ViT tokens flip when
               nothing at that tile changed (PROPOSAL.md s6 measured this cost for 256-code tile targets)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import teval as T  # noqa: E402
from tworld import POOLS, quantize  # noqa: E402

OUT = ROOT / "artifacts/eda/levers_codebooks_v1"


@torch.no_grad()
def kmeans(x, k, device, iters=30):
    g = torch.Generator().manual_seed(0)
    codes = x[torch.randperm(len(x), generator=g)[:k]].clone().to(device)
    xd = x.to(device)
    for _ in range(iters):
        idx = torch.cat([torch.cdist(xd[i:i + 65536], codes).argmin(-1) for i in range(0, len(xd), 65536)])
        sums = torch.zeros_like(codes).index_add_(0, idx, xd)
        counts = torch.bincount(idx, minlength=k).float()
        empty = counts == 0
        codes = torch.where(empty[:, None], codes, sums / counts.clamp_min(1)[:, None])
        if empty.any():                                   # re-seed empty clusters from random tokens
            codes[empty] = xd[torch.randint(len(xd), (int(empty.sum()),), generator=g).to(device)]
    return codes.cpu(), int(empty.sum())


def sample_tokens(pool_name, n=500_000):
    pool = torch.load(POOLS[pool_name] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
    g = torch.Generator().manual_seed(0)
    pick = rows[torch.randperm(len(rows), generator=g)[:4000]].sort().values
    toks = pool["tokens"][pick].float().flatten(0, 2)                           # 4000*6*81 = 1.94M
    return toks[torch.randperm(len(toks), generator=g)[:n]]


def main():
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    result = {}
    for pool_name in sys.argv[1:] or ["raw"]:
        cache = T.build_cache(pool_name, device)
        probes = T.Probes(cache, meta, train_roots, train_seeds)
        x = sample_tokens(pool_name)
        fut = cache["fut"][test].float()                                           # [n,16,81,192]
        fv = meta["future_visible"][test, 0]
        mean = x.mean(0)
        cont = probes.read(fut[:, 7], fv[:, 7])
        # unchanged pairs: all 63 tiles and zombie/cow grids identical between frames k and k+1
        vis = fv.float()
        tiles = vis[..., :1071].reshape(*vis.shape[:2], 63, 17).argmax(-1)
        mobs = vis[..., 1071:1512].reshape(*vis.shape[:2], 63, 7)[..., :3]
        same = (tiles[:, 1:] == tiles[:, :-1]).all(-1) & (mobs[:, 1:] == mobs[:, :-1]).all(-1).all(-1)
        for k in (1024, 4096):
            codes, empty = kmeans(x, k, device)
            torch.save({"codes": codes, "pool": pool_name, "K": k, "empty_after_last_iter": empty}, OUT / f"{pool_name}_K{k}.pt")
            q = codes[quantize(fut.to(device), codes.to(device)).cpu()]
            rel = float(((q - fut) ** 2).sum(-1).mean() / ((fut - mean) ** 2).sum(-1).mean())
            idx = quantize(fut[:, :, T.MAP].to(device), codes.to(device)).cpu()     # [n,16,63]
            flip = float((idx[:, 1:] != idx[:, :-1])[same].float().mean())
            result[f"{pool_name}_K{k}"] = {"rel_error": rel, "flip_rate_unchanged": flip, "unchanged_pairs": int(same.sum()),
                                           "facts_quantized_k8": probes.read(q[:, 7], fv[:, 7]),
                                           "facts_continuous_k8": cont, "empty_clusters": empty}
            print(pool_name, k, json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in result[f"{pool_name}_K{k}"].items()
                                           if not kk.startswith("facts")}), flush=True)
            print("   facts quantized", json.dumps({a: round(b, 3) for a, b in result[f"{pool_name}_K{k}"]["facts_quantized_k8"].items()}))
            print("   facts continuous", json.dumps({a: round(b, 3) for a, b in cont.items()}), flush=True)
    (HERE / "codebook.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
