"""D20. Does canonical H2's world fail because it was trained with BatchNorm batch statistics and is deployed with
running statistics?

LeWM has two BatchNorm1d layers: the encoder projector's (z itself) and the world's predictor projector's
(its output). In joint training (modes recorded in the checkpoint) both run in TRAIN mode, i.e. with the
statistics of each 128-window x 4-frame batch. Everything after joint -- the latent cache (cache.py: encoder
frozen = eval), the bridge (TC-15 freezes the predictor BN), the gate, every diagnostic -- uses running
statistics.

Joint-like batches: 16 batches x 128 windows of 4 consecutive frames + 3 actions, uniform over TRAIN-split
transitions of the M4 corpus (seed 0). Per batch, z is computed in both encoder modes; each world predicts
frames 1-3 from 0-2 (teacher, the joint objective) with its predictor BN in both modes. BN momentum is set to
0 so no pass changes the running statistics. Worlds: joint step-10000 (pre-bridge) and bridge step-2000.
Reported: prediction MSE per element and copy-previous-frame MSE in the same z; ratio < 1 beats copying;
and the size of the train/eval z difference relative to z's variance and to the one-step change.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from insample import STORES  # noqa: E402

BATCHES, B, T = 16, 128, 4


def windows(seed):
    index = []
    for store in STORES:
        for s in sorted(store.glob("shard-*.pt")):
            for j, e in enumerate(torch.load(s, weights_only=False, mmap=True)["episodes"]):
                if e["split"] == "train" and len(e["actions_taken"]) >= T:
                    index.append((s, j, len(e["actions_taken"])))
    lengths = torch.tensor([n - T + 2 for _, _, n in index], dtype=torch.float64)
    gen = torch.Generator().manual_seed(seed)
    picks = torch.multinomial(lengths, BATCHES * B, replacement=True, generator=gen)
    items = []
    for p in picks.tolist():
        s, j, n = index[p]
        items.append((s, j, int(torch.randint(n - T + 2, (), generator=gen))))
    frames, actions = [None] * len(items), [None] * len(items)
    by_shard = {}
    for k, (s, j, t) in enumerate(items):
        by_shard.setdefault(s, []).append((k, j, t))
    for s, rows in by_shard.items():
        eps = torch.load(s, weights_only=False, mmap=True)["episodes"]
        for k, j, t in rows:
            frames[k] = torch.as_tensor(np.asarray(eps[j]["observations"][t:t + T]))
            actions[k] = eps[j]["actions_taken"][t:t + T - 1].clone()
    return torch.stack(frames), torch.stack(actions)


def load(path, phase):
    from d4mj.checkpoint import read_lewm_bridge, read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    payload = (read_lewm_bundle if phase == "joint" else read_lewm_bridge)(path)
    bundle = ModelBundle.create(config_from_dict(payload["config"]))
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    bundle.world.load_state_dict(payload["modules"]["world"])
    for m in list(bundle.encoder.modules()) + list(bundle.world.modules()):
        if isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = 0.0
    bundle.encoder.requires_grad_(False); bundle.world.requires_grad_(False)
    return bundle


@torch.no_grad()
def main():
    device = torch.device("cuda")
    frames, actions = windows(0)
    joint = load(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt", "joint")
    bridge = load(ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt", "bridge")
    enc = joint.encoder.to(device)
    result = {"z": {}, "worlds": {}}
    acc = {}
    zdiff, zvar, zstep = [], [], []
    for b in range(BATCHES):
        f = frames[b * B:(b + 1) * B].to(device)
        a = actions[b * B:(b + 1) * B].to(device)
        z = {}
        for mode in ("train", "eval"):
            enc.train(mode == "train")
            with torch.autocast(device_type="cuda", enabled=False):
                z[mode] = enc(f).float()                                   # [B, T, 1, 192]
        enc.eval()
        zdiff.append(((z["train"] - z["eval"]) ** 2).mean().item())
        zvar.append(z["eval"].flatten(0, 2).var(0).mean().item())
        zstep.append(((z["eval"][:, 1:] - z["eval"][:, :-1]) ** 2).mean().item())
        for wname, bundle in (("joint", joint), ("bridge", bridge)):
            world = bundle.world.to(device)
            for zmode in ("train", "eval"):
                for pmode in ("train", "eval"):
                    world.eval()
                    world.predictor_projector.train(pmode == "train")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = world.teacher(z[zmode], a).predicted.float()
                    target = z[zmode][:, 1:]
                    key = (wname, zmode, pmode)
                    acc.setdefault(key, [[], []])
                    acc[key][0].append(((pred - target) ** 2).mean().item())
                    acc[key][1].append(((z[zmode][:, :-1] - target) ** 2).mean().item())
            world.eval()
    result["z"] = {"train_eval_z_msd": float(np.mean(zdiff)), "z_variance_per_element": float(np.mean(zvar)),
                   "one_step_change_msd": float(np.mean(zstep))}
    print(json.dumps(result["z"]), flush=True)
    for (wname, zmode, pmode), (e, c) in acc.items():
        row = {"prediction_mse": float(np.mean(e)), "copy_mse": float(np.mean(c)), "ratio": float(np.mean(e) / np.mean(c))}
        result["worlds"][f"{wname}/z_{zmode}/predictorBN_{pmode}"] = row
        print(f"{wname:6s} z {zmode:5s} predictorBN {pmode:5s}", json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
    (HERE / "bnmode.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
