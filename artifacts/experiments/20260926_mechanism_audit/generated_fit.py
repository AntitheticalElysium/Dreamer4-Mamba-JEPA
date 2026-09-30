"""Readout-transfer control for the 18k corrt world on diagnostic futures.

Train identical ridge probes on true or imagined TRAIN-seed frames. Compare
both on imagined TEST-seed frames at fixed rollout depths. The 70/30 seed
split, validation subset, labels, and probe capacity come from teval.py.
This tests decoder transfer, not action selection or physical fidelity.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
LEVERS = ROOT / "artifacts/experiments/20260927_levers"
sys.path.insert(0, str(LEVERS))
import teval  # noqa: E402


def main():
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    meta, train_roots, train_seeds = teval.split()
    cache = teval.build_cache("raw", device)
    path = ROOT / "artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_u18000.pt"
    world, st = teval.load_world(path, device)
    from d4mj.config import config_from_dict
    import spatial as S
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    R, H = len(meta["seed"]), teval.H
    gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
    for i in range(0, R, 8):
        ctx = cache["ctx"][i:i + 8].float()
        ca = cache["ctx_a"][i:i + 8]
        fa = cache["fut_a"][i:i + 8]
        g_frames = [ctx[:, j] for j in range(4)]
        a_hist = [ca[:, j] for j in range(3)]
        for k in range(H):
            w = 4 if k == 0 else 5
            a = torch.stack(a_hist[-(w - 1):] + [fa[:, k]], 1)
            g = teval.step(world, torch.stack(g_frames[-w:], 1), a, device, config)
            gen[i:i + len(ctx), k] = g.half()
            g_frames.append(g)
            a_hist.append(fa[:, k])
    del world
    torch.cuda.empty_cache()
    print("imagined frames collected", flush=True)
    true_probe = teval.Probes(cache, meta, train_roots, train_seeds)
    print("true-fit probes ready", flush=True)
    generated_cache = {"ctx": cache["ctx"], "fut": gen}
    generated_probe = teval.Probes(generated_cache, meta, train_roots, train_seeds)
    print("generated-fit probes ready", flush=True)
    test_roots = ~train_roots
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    fv = meta["future_visible"][:, 0]
    out = {"world": st["name"], "roots": R,
           "train_seeds": int(len(train_seeds)), "test_roots": int(test_roots.sum()),
           "method": "same teval.Probes ridge, train on true versus imagined TRAIN seeds; test on imagined TEST seeds",
           "depth": {}}
    for k in (1, 4, 8, 16):
        m = test_roots & alive[:, k - 1]
        x = gen[m, k - 1].float()
        y = fv[m, k - 1]
        out["depth"][k] = {"n": int(m.sum()), "true_fit": true_probe.read(x, y),
                            "generated_fit": generated_probe.read(x, y)}
        print(f"depth {k}: " + json.dumps(out["depth"][k]), flush=True)
        if k == 16:
            from compound import auc
            label = teval.facts_of(y)["zombie"][:, list(teval.NEAR)]
            scores = {"true_fit": true_probe.zombie(x[:, list(teval.NEAR)].flatten(0, 1)).view(-1, 4),
                      "generated_fit": generated_probe.zombie(x[:, list(teval.NEAR)].flatten(0, 1)).view(-1, 4)}
            seeds = meta["seed"][m]
            groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
            rng = torch.Generator().manual_seed(20260928)
            draws = []
            for _ in range(1000):
                chosen = torch.randint(len(groups), (len(groups),), generator=rng)
                idx = torch.cat([groups[j] for j in chosen])
                a = auc(scores["true_fit"][idx], label[idx])
                b = auc(scores["generated_fit"][idx], label[idx])
                draws.append([a, b, b - a])
            d = torch.tensor(draws)
            out["depth16_bootstrap"] = {"unit": "walk seed", "seeds": len(groups), "draws": len(draws),
                                        "ci95": {name: [float(q) for q in d[:, j].quantile(torch.tensor([0.025, 0.975]))]
                                                 for j, name in enumerate(("true_fit", "generated_fit", "generated_minus_true"))}}
    dest = Path(__file__).with_suffix(".json")
    dest.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
