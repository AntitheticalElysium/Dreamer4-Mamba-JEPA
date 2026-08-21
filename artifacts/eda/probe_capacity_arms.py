"""Probe each bottleneck arm's exported Z* and pre-bottleneck features.

Same roots, same whole-seed split and the same probe as everywhere else. The 64 arm
exports twice as many scalars, so its probe also gets twice the input width; a
random projection of Z*_64 down to 512 is reported beside it so the primary
comparison can be read with probe width held equal as well as free.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval
from probe_observability import seed_split
from probe_prebottleneck import fit_probe

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.representation import Encoder, pack

DEVICE = "cuda"


@torch.no_grad()
def encode(path: Path, n_latents: int, d_bottleneck: int, rows):
    from dataclasses import replace

    # `checkpoint.load` compares the whole config, so the arm must be restored under
    # the exact config it was written with -- batch and seed included.
    report = json.loads((path.parent / "training_report.json").read_text())
    config = replace(Config(), n_latents=n_latents, d_bottleneck=d_bottleneck,
                     batch=report["batch"], seed=report["seed"])
    encoder = Encoder(config).to(DEVICE)
    captured: dict[str, torch.Tensor] = {}
    encoder.bottleneck.register_forward_pre_hook(
        lambda m, i: captured.__setitem__("pre", i[0].detach()))
    from d4mj.representation import Decoder

    load(path, config, part0=encoder, part1=Decoder(config))
    encoder.eval()
    # roots near an episode start carry a shorter history; padding them would change
    # their causal context, so each length is encoded in its own batch and the
    # original order restored afterwards.
    by_length: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_length.setdefault(int(row["frames"].shape[0]), []).append(index)
    pre = [None] * len(rows)
    z = [None] * len(rows)
    for length, indices in sorted(by_length.items()):
        for lo in range(0, len(indices), 8):
            chunk = indices[lo : lo + 8]
            frames = torch.stack([rows[i]["frames"] for i in chunk])
            out, _, _ = encoder(patchify(frames, config.patch).to(DEVICE))
            block_pre = captured["pre"][:, -1].reshape(len(chunk), -1).cpu().half()
            block_z = pack(out, config)[:, -1].reshape(len(chunk), -1).cpu()
            for position, index in enumerate(chunk):
                pre[index], z[index] = block_pre[position], block_z[position]
    return torch.stack(pre).float(), torch.stack(z)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", type=int, default=3000)
    parser.add_argument("--arm", type=str, default=None,
                        help="probe one arm per process; GPU memory is not released "
                             "between arms in one process (the S44 lesson)")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--suffix", type=str, default="s0",
                        help="tokenizer-seed suffix of the arm folders")
    args = parser.parse_args()

    rows = []
    for path in sorted((HERE / "root_frames").glob("shard-*.pt")):
        rows += torch.load(path, weights_only=False)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    labels = torch.stack([r["label"] for r in rows])
    print(f"{len(rows)} damage roots  fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}")

    root = HERE / ("capacity6k" if (HERE / "capacity6k").exists() else "capacity")
    arms = {"32x16": (32, 16, root / f"n32d16_{args.suffix}"),
            "64x16": (64, 16, root / f"n64d16_{args.suffix}")}
    store = root / f"values_{args.suffix}_{args.milestone:06d}"
    store.mkdir(parents=True, exist_ok=True)
    if args.combine:
        values = {}
        for path in sorted(store.glob("*.pt")):
            blob = torch.load(path, weights_only=False)
            values.update({k: (v["values"], v["keep"]) for k, v in blob.items()})
        base_v, base_k = values["32x16/Z*"]
        print(f"\ncombined, milestone {args.milestone}")
        for name, (v, k) in sorted(values.items()):
            a, (lo, hi) = interval(v, 17)
            print(f"  {name:<26}{a:>9.4f}  [{lo:.4f}, {hi:.4f}]")
        print()
        targets = ["64x16/Z*"] + sorted(k for k in values if "rp512" in k) + \
                  ["64x16/pre-bottleneck"]
        for name in targets:
            if name not in values:
                continue
            v, k = values[name]
            both = base_k & k
            d, (lo, hi) = interval(v[both[k]] - base_v[both[base_k]], 23)
            star = "   <-- PRIMARY" if name == "64x16/Z*" else ""
            print(f"  paired {name:<22} minus 32x16/Z*: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
        return
    if args.arm:
        arms = {args.arm: arms[args.arm]}
    results, values = {}, {}
    for name, (n_latents, d_bottleneck, folder) in arms.items():
        checkpoint = folder / f"encoder_{args.milestone:06d}.pt"
        if not checkpoint.exists():
            print(f"  {name}: {checkpoint.name} missing, skipped")
            continue
        pre, z = encode(checkpoint, n_latents, d_bottleneck, rows)
        row = {}
        for label, x, seed in (("pre-bottleneck", pre, 6), ("Z*", z, 2)):
            v, keep, tune = fit_probe(x, labels, splits, seed)
            values[f"{name}/{label}"] = (v, keep)
            a, (lo, hi) = interval(v, 17)
            row[label] = {"within_auc": a, "ci": [lo, hi], "tune": tune, "dim": x.shape[1]}
            print(f"  {name:<7}{label:<16}{a:>9.4f}  [{lo:.4f}, {hi:.4f}]  dim {x.shape[1]}")
        if z.shape[1] > 512:
            # several fixed projections, so the width-matched control is not one draw
            for index, projection_seed in enumerate((20260820, 7, 101, 4242, 31337)):
                generator = torch.Generator().manual_seed(projection_seed)
                projection = (torch.randn(z.shape[1], 512, generator=generator)
                              / z.shape[1] ** 0.5)
                v, keep, tune = fit_probe(z @ projection, labels, splits, 12 + index)
                values[f"{name}/Z*-rp512-s{projection_seed}"] = (v, keep)
                a, (lo, hi) = interval(v, 17)
                row[f"Z*-rp512-s{projection_seed}"] = {"within_auc": a, "ci": [lo, hi]}
                print(f"  {name:<7}{f'Z*->512 rp seed {projection_seed}':<22}"
                      f"{a:>9.4f}  [{lo:.4f}, {hi:.4f}]")
        row["delta_bottleneck"] = row["pre-bottleneck"]["within_auc"] - row["Z*"]["within_auc"]
        results[name] = row

    torch.save({k: {"values": v[0], "keep": v[1]} for k, v in values.items()},
               store / f"{list(arms)[0]}.pt")
    if "32x16/Z*" in values and "64x16/Z*" in values:
        print()
        base_v, base_k = values["32x16/Z*"]
        targets = ["64x16/Z*"] + sorted(k for k in values if "rp512" in k) + \
                  ["64x16/pre-bottleneck"]
        for name in targets:
            if name not in values:
                continue
            v, k = values[name]
            both = base_k & k
            d, (lo, hi) = interval(v[both[k]] - base_v[both[base_k]], 23)
            star = "  <-- PRIMARY" if name == "64x16/Z*" else ""
            print(f"  paired {name:<24} minus 32x16/Z*: {d:+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")
            results.setdefault("paired", {})[name] = {"delta": d, "ci": [lo, hi]}
    (root / f"probe_{args.suffix}_{args.milestone:06d}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
