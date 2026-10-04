"""Post hoc, fifth step: is the forecast written into the generated HEALTH TILE, or only elsewhere?

spatial_why4 (reading `mixed`): one-step damage is forecastable from the real frame t + action (DEV
AUC 0.909) and TH's predictor hidden state holds it (0.896). Its `generated_TH` input (frame t tokens
+ generated t+1 tokens, 0.911) could not isolate the consequence: frame t shows the cause and the
generated successor reveals the action through its movement, so the pair predicts damage without the
generated health tile ever changing -- which spatial_why3 found it never does. That test was
mis-specified; this one isolates the tile.

Same TRAIN / DEV transitions and probe recipe as spatial_why4 (balanced BCE, 3,000 steps, three
seeds, DEV AUC of P(damage)). Inputs, no action given to any:
  real_tile        the REAL successor's health tile token (63) with frame t's      hindsight control
  generated_tile   TH's GENERATED successor's health tile token with frame t's
  generated_alone  TH's generated successor, all 81 tokens, without frame t
  real_alone       the real successor, all 81 tokens, without frame t              hindsight control
Tile inputs use an MLP (384 -> 512 -> 1); full-token inputs use spatial_why4's attention probe.

Readings (committed before the run; post hoc):
  real_tile < 0.95                                   -> void (the tile does not show the outcome)
  generated_tile < 0.65                              -> not_written_to_tile
  and generated_alone >= 0.85 besides                -> carried_elsewhere_not_in_tile
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

from spatial import TOKENS, WIDTH, WORLDS, World, bridge  # noqa: E402
from spatial_why2 import dev_population  # noqa: E402
from spatial_why4 import SEED, STEPS, Probe, auc, train_transitions, world_features  # noqa: E402

TILE = 63


def fit(x_train, y_train, x_dev, device, seed):
    torch.manual_seed(seed)
    tile = x_train.ndim == 2
    probe = (nn.Sequential(nn.Linear(x_train.shape[-1], 512), nn.ReLU(), nn.Linear(512, 1)) if tile
             else Probe(x_train.shape[-1], False)).to(device)
    head = (lambda x: probe(x)[:, 0]) if tile else probe
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    pos = ((1 - y_train).sum() / y_train.sum().clamp_min(1)).to(device)
    for _ in range(STEPS):
        idx = torch.randint(len(x_train), (256,), generator=gen)
        loss = F.binary_cross_entropy_with_logits(head(x_train[idx].float().to(device)), y_train[idx].to(device),
                                                  pos_weight=pos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        return torch.cat([head(x_dev[i:i + 512].float().to(device)).cpu() for i in range(0, len(x_dev), 512)])


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    gen = torch.Generator().manual_seed(SEED)
    encoder, config = bridge()
    dev = dev_population(encoder, config, device, gen, log)
    del encoder
    torch.cuda.empty_cache()
    train = train_transitions(gen)
    y_train, y_dev = (train["dh"] <= -2).float(), (dev["dh"] <= -2).float()
    stored = torch.load(WORLDS / "TH.pt", map_location="cpu", weights_only=False)
    world = World(TOKENS, True).to(device)
    world.load_state_dict(stored["world"])
    world.eval()
    _, g_train = world_features(world, config, train, device)
    _, g_dev = world_features(world, config, dev, device)
    del world
    torch.cuda.empty_cache()
    pair = lambda now, then: torch.cat([now[:, TILE], then[:, TILE]], -1)
    inputs = {"real_tile": (pair(train["tokens"][:, 3], train["tokens"][:, 4]), pair(dev["tokens"][:, 3], dev["tokens"][:, 4])),
              "generated_tile": (pair(train["tokens"][:, 3], g_train), pair(dev["tokens"][:, 3], g_dev)),
              "generated_alone": (g_train, g_dev),
              "real_alone": (train["tokens"][:, 4], dev["tokens"][:, 4])}
    result = {}
    for name, (xt, xd) in inputs.items():
        aucs = [auc(y_dev, fit(xt, y_train, xd, device, s)) for s in range(3)]
        result[name] = {"dev_auc": float(np.mean(aucs)), "per_seed": aucs}
        log(input=name, dev_auc=round(result[name]["dev_auc"], 4))
    r, g, a = (result[k]["dev_auc"] for k in ("real_tile", "generated_tile", "generated_alone"))
    reading = ("void" if r < 0.95 else
               ("carried_elsewhere_not_in_tile" if a >= 0.85 else "not_written_to_tile") if g < 0.65 else "mixed")
    evidence = {"schema": "d4mj_spatial_why5_v1", "status": "POST HOC", "script_sha256": _sha256(Path(__file__)),
                "reading": reading, "result": result}
    (HERE / "evidence/spatial_why5.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="spatial_why5_complete", reading=reading)


if __name__ == "__main__":
    main()
