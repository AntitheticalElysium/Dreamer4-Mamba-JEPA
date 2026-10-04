"""D21. Canonical H2's one-step quality as a function of how much history its Mamba state has consumed.

Joint training (config joint.frames = 4) trains the world on 4-frame windows from a zero state. The bridge
trains on 32/128-frame windows after a burn-in of ~50-70 frames (bridge metrics.jsonl `burn_lengths`). The
readout-ladder evaluator uses 4 frames from a zero state; the H2 gate uses the bridge's burn-in.
Also the interface Mamba worlds Z and U (interface.py: phase 1 on 4-frame, phase 2 on 6-frame windows), on
the same frames (U: the whitened pool's PCA of the encoder's own 4x4 grid).
Here: 256 windows of 96 consecutive frames (TRAIN split, seed 2; eval-mode z, fp32 encoder), teacher-forced
from a zero state over all 96 frames. Prediction MSE / copy-previous-frame MSE at each position, binned:
[1-3] [4-7] [8-15] [16-31] [32-63] [64-95] -- for the joint (pre-bridge) and the bridge world.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import bnmode  # noqa: E402

BINS = ((1, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 95))


@torch.no_grad()
def main():
    device = torch.device("cuda")
    bnmode.BATCHES, bnmode.B, bnmode.T = 8, 32, 96
    frames, actions = bnmode.windows(2)
    worlds = {"joint": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt", "joint"),
              "bridge": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt", "bridge")}
    enc = worlds["joint"].encoder.to(device).eval()
    sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
    import interface as I
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    _, config = I.load_bridge()
    for name, path in (("Z", "interface_worlds_v1/Z.pt"), ("U", "interface_worlds_v1/U.pt")):
        b = I.world_bundle(config, enc, device)
        b.world.load_state_dict(torch.load(ROOT / "artifacts/eda" / path, map_location="cpu", weights_only=False)["world"])
        worlds[name] = b
    err = {w: [] for w in worlds}
    copy_u = []
    copy = []
    for i in range(0, len(frames), 32):
        f, a = frames[i:i + 32].to(device), actions[i:i + 32].to(device)
        with torch.autocast(device_type="cuda", enabled=False):
            parts = [enc.export(f[j:j + 4], grid=4) for j in range(0, len(f), 4)]
        z = torch.cat([p[0] for p in parts]).float()
        grid = torch.cat([p[2] for p in parts]).float().flatten(2)
        u = I.state_of("U", pool["pca"], z[:, :, 0].cpu(), grid.cpu()).to(device)[:, :, None]
        copy.append(((z[:, :-1] - z[:, 1:]) ** 2).mean((0, 2, 3)).cpu())
        copy_u.append(((u[:, :-1] - u[:, 1:]) ** 2).mean((0, 2, 3)).cpu())
        for w, b in worlds.items():
            world = b.world.to(device).eval()
            s = u if w == "U" else z
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = world.teacher(s, a).predicted.float()
            err[w].append(((pred - s[:, 1:]) ** 2).mean((0, 2, 3)).cpu())      # per predicted position 1..95
    copy = torch.stack(copy).mean(0)
    copy_u = torch.stack(copy_u).mean(0)
    result = {}
    for w in worlds:
        e = torch.stack(err[w]).mean(0)
        c = copy_u if w == "U" else copy
        result[w] = {f"{lo}-{hi}": float(e[lo - 1:hi].mean() / c[lo - 1:hi].mean()) for lo, hi in BINS}
        print(w, json.dumps({k: round(v, 3) for k, v in result[w].items()}), flush=True)
    result["copy_mse_by_bin"] = {f"{lo}-{hi}": float(copy[lo - 1:hi].mean()) for lo, hi in BINS}
    (HERE / "context.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
