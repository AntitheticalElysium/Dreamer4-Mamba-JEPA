"""Frozen-canvas intervention: remove absent steps from both Mamba convolution and SSM.

Post-hoc mechanism test on the already-inspected diagnosis TEST block. This
changes inference only, preserving the checkpoint, actions and generated inputs.
It tests the effect of the original masked_scan's incomplete hold; it is not a
trained corrected-canvas result or a new gate.
"""
import hashlib
import json
from pathlib import Path

import torch

import teval as T
import tworld as TW

HERE = Path(__file__).parent
WORLD = Path("artifacts/eda/levers_tworlds_v1/corrt_raw_suffix_s7_fcanvas.pt")


def exact_hold(mixer, inputs, keep):
    """Reference: scan only kept inputs for each cell; scatter outputs back."""
    n, length, dim = inputs.shape
    result = inputs.new_zeros(n, length, dim)
    pattern = (keep.long() * (2 ** torch.arange(length, device=keep.device))).sum(-1)
    for code in pattern.unique().tolist():
        if code == 0:
            continue
        rows = torch.where(pattern == code)[0]
        steps = torch.where(keep[rows[0]])[0]
        y, _ = mixer(inputs[rows][:, steps])
        result[rows[:, None], steps[None, :]] = y.to(result.dtype)
    return result


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S

    device = torch.device("cuda")
    meta, train_roots, _ = T.split()
    ix = torch.where(~train_roots)[0]
    cache = T.build_cache("raw", device)
    world, payload = T.load_world(WORLD, device)
    source_sha = hashlib.sha256(Path(TW.__file__).read_bytes()).hexdigest()
    assert payload["script_sha256"] == source_sha
    ckpt_sha = hashlib.sha256(WORLD.read_bytes()).hexdigest()
    native = torch.load(HERE / "canvas_oracle_per_root.pt", weights_only=False)
    assert torch.equal(native["root_index"], ix)
    assert torch.equal(native["seed"], meta["seed"][ix])
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])

    # Ensure the replacement agrees with the original when no cell is absent.
    x = torch.randn(9, 5, world.layers[0].mix.core.d_model, device=device, generator=torch.Generator(device=device).manual_seed(7))
    m = world.layers[0].mix
    all_kept = torch.ones(9, 5, dtype=torch.bool, device=device)
    all_kept_maxabs = float((exact_hold(m, x, all_kept) - TW.masked_scan(m, x, all_kept)).abs().max())
    assert all_kept_maxabs < 1e-4, all_kept_maxabs

    original = TW.masked_scan
    TW.masked_scan = exact_hold
    pred = torch.empty(len(ix), T.H, 81, 192, dtype=torch.float16)
    try:
        for i in range(0, len(ix), 4):
            sl = ix[i:i + 4]
            ctx = cache["ctx"][sl].float()
            ca, fa = cache["ctx_a"][sl], cache["fut_a"][sl]
            frames = [ctx[:, j] for j in range(4)]
            actions = [ca[:, j] for j in range(3)]
            for k in range(T.H):
                window = 4 if k == 0 else 5
                a = torch.stack(actions[-(window - 1):] + [fa[:, k]], 1)
                nxt = T.step(world, torch.stack(frames[-window:], 1), a, device, config)
                pred[i:i + len(sl), k] = nxt.half()
                frames.append(nxt)
                actions.append(fa[:, k])
    finally:
        TW.masked_scan = original
    fut = cache["fut"][ix].float()
    err = ((pred.float() - fut) ** 2).sum((-1, -2))
    alive, seeds, V = native["alive"], native["seed"], native["V"]
    out = {"checkpoint_sha256": ckpt_sha, "model_source_sha256": source_sha,
           "reference": "compressed-kept convolution and SSM; frozen checkpoint, inference only",
           "all_kept_maxabs": all_kept_maxabs, "n_test_roots": len(ix),
           "walk_seeds": int(seeds.unique().numel()), "depth": {}}
    for k in (1, 4, 8, 16):
        mask = alive[:, k - 1]
        a = float(native["native_err"][mask, k - 1].mean() / V)
        b = float(err[mask, k - 1].mean() / V)
        out["depth"][k] = {"n": int(mask.sum()), "native": a, "full_hold": b,
                           "difference": b - a, "relative_change": (b - a) / a}
    groups = [torch.where(seeds == seed)[0] for seed in seeds.unique()]
    g = torch.Generator().manual_seed(20260929)
    ds = []
    for _ in range(2000):
        rows = torch.cat([groups[j] for j in torch.randint(len(groups), (len(groups),), generator=g)])
        mask = alive[rows, 15]
        a = float(native["native_err"][rows, 15][mask].mean() / V)
        b = float(err[rows, 15][mask].mean() / V)
        ds.append(b - a)
    out["depth16_difference_ci95"] = [float(x) for x in torch.tensor(ds).quantile(torch.tensor([.025, .975]))]
    torch.save({"native_err": native["native_err"], "full_hold_err": err,
                "alive": alive, "seed": seeds, "root_index": ix, "V": V},
               HERE / "canvas_hold_oracle_per_root.pt")
    (HERE / "canvas_hold_oracle.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
