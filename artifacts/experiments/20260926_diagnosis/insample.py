"""D18. Is the one-step failure present on the worlds' OWN training distribution, or only off it?

Transitions sampled uniformly from the M4 corpus (expert_v1 + support_v2), 4,000 from TRAIN-split episodes
and 4,000 from DEV/FINAL-split episodes (sampling seed 0), plus the diagnostic futures' factual depth-1
transitions (D9). Each: 4 context frames t-3..t, their 3 actions, the action a_t, the true frame t+1.
Class of a_t (pixels only): moves -> moved / blocked / ambiguous (coverage.classify, 98.8% accurate when it
decides); SLEEP onset (sleep.asleep: t awake, t+1 asleep); everything else -> other.
Per world, per split and class: one-step prediction (evaluator protocol: teacher on the 4 frames, one advance;
transformers: the 4-frame window) squared error / copying the root's squared error. < 1 beats copying.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER)); sys.path.insert(0, str(HERE))
from coverage import classify  # noqa: E402
from rollouts import encode, load_roots, load_worlds, states  # noqa: E402
from sleep import asleep  # noqa: E402

STORES = (ROOT / "artifacts/craftax_expert_store_v1", ROOT / "artifacts/craftax_support_v2")
PER_SPLIT, N = 4000, 17
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")


def sample(split_names, count, seed):
    """Uniform over transitions t in [3, len-1) of episodes whose split is in split_names."""
    index = []
    for store in STORES:
        for s in sorted(store.glob("shard-*.pt")):
            for j, e in enumerate(torch.load(s, weights_only=False, mmap=True)["episodes"]):
                if e["split"] in split_names and len(e["actions_taken"]) >= 5:
                    index.append((s, j, len(e["actions_taken"])))
    lengths = torch.tensor([n - 3 for _, _, n in index], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    picks = torch.multinomial(lengths, count, replacement=True, generator=gen)
    by_shard = {}
    for p in picks.tolist():
        s, j, n = index[p]
        t = 3 + int(torch.randint(n - 3, (), generator=gen))
        by_shard.setdefault(s, []).append((j, t))
    out = {"frames": [], "ctx_a": [], "a": [], "next": []}
    for s, items in by_shard.items():
        eps = torch.load(s, weights_only=False, mmap=True)["episodes"]
        for j, t in items:
            e = eps[j]
            obs, act = e["observations"], e["actions_taken"]
            out["frames"].append(torch.as_tensor(np.asarray(obs[t - 3:t + 1])))
            out["ctx_a"].append(act[t - 3:t].clone()); out["a"].append(int(act[t]))
            out["next"].append(torch.as_tensor(np.asarray(obs[t + 1])))
    return {"frames": torch.stack(out["frames"]), "ctx_a": torch.stack(out["ctx_a"]),
            "a": torch.tensor(out["a"]), "next": torch.stack(out["next"])}


def futures():
    roots = load_roots()
    return {"frames": torch.stack([r["context"] for r in roots]), "ctx_a": torch.stack([r["context_actions"] for r in roots]),
            "a": torch.stack([r["future_actions"][0] for r in roots]), "next": torch.stack([r["future_frames"][0, 0] for r in roots])}


def classes(d):
    a, before, after = d["a"], d["frames"][:, -1], d["next"]
    cls = ["other"] * len(a)
    move = (a >= 1) & (a <= 4)
    if move.any():
        got = classify(before[move], after[move], a[move])
        for i, g in zip(torch.where(move)[0].tolist(), got.tolist()):
            cls[i] = {1: "moved", -1: "blocked", 0: "ambiguous"}[g]
    onset = (~asleep(before)) & asleep(after)
    for i in torch.where(onset)[0].tolist():
        cls[i] = "sleep_onset"
    return cls


@torch.no_grad()
def predict(w, world, config, s_ctx, ctx_a, a, device):
    from d4mj.train import autocast_context
    with autocast_context(config):
        if w in ("sZ", "T"):
            x = s_ctx[:, :, None] if w == "sZ" else s_ctx
            acts = torch.cat([ctx_a, a[:, None]], 1).to(device)
            out = world(x.to(device), acts)[0][:, -1].float().cpu()
            return out[:, 0] if w == "sZ" else out
        st = world.world.teacher(s_ctx[:, :, None].to(device), ctx_a.to(device)).state
        adv, _ = world.advance(st, a[:, None].to(device))
        return adv.latent[:, 0, 0].float().cpu()


def main():
    device = torch.device("cuda")
    sys.path.insert(0, str(LADDER))
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    encoder, config, worlds = load_worlds(device)
    data = {"train": sample({"train"}, PER_SPLIT, 0), "heldout": sample({"dev", "final"}, PER_SPLIT, 1), "futures": futures()}
    result = {}
    for split, d in data.items():
        cls = classes(d)
        n = len(d["a"])
        ctx = [states(pool, *[x.cpu() for x in encode(encoder, d["frames"][:, i], device)]) for i in range(4)]
        nxt = states(pool, *[x.cpu() for x in encode(encoder, d["next"], device)])
        result[split] = {"n": n, "class_counts": {c: cls.count(c) for c in sorted(set(cls))}, "worlds": {}}
        for w in WORLDS:
            s_ctx = torch.stack([c[w] for c in ctx], 1)
            pred = torch.cat([predict(w, worlds[w], config, s_ctx[i:i + 64], d["ctx_a"][i:i + 64], d["a"][i:i + 64], device)
                              for i in range(0, n, 64)])
            true, root = nxt[w].float(), s_ctx[:, -1].float()
            err = ((pred - true) ** 2).flatten(1).sum(1)
            cp = ((root - true) ** 2).flatten(1).sum(1)
            row = {"all": float(err.mean() / cp.mean())}
            for c in sorted(set(cls)):
                m = torch.tensor([x == c for x in cls])
                if m.sum() >= 30:
                    row[c] = float(err[m].mean() / cp[m].mean())
            result[split]["worlds"][w] = row
            print(split, w, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
        print(split, "classes", json.dumps(result[split]["class_counts"]), flush=True)
    (HERE / "insample.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
