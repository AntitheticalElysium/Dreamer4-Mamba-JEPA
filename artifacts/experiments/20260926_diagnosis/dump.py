"""D2. Dump every trained world's one-step readouts on the opened blocks (55k-58k), opportunity roots only.

Per root and action: the head's P(dead) on the GENERATED successor and on the TRUE successor (stored key-1
frame, `observe_latent`: same Mamba history h, latent replaced by truth), the generated latent, and h (the
Mamba output after consuming (context, action), from which the latent is projected). Plus the root state.
Evaluator protocol unchanged (interface.branches): 4 observed frames through teacher, one advance per action.
Worlds: canonical Raw H2 (its own heads), and the interface worlds Z, ZW, U, W (seed 1) and W seed 2.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(LADDER))
import interface as I  # noqa: E402

OUT = ROOT / "artifacts/eda/diagnosis_dump_v1"
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}
N = 17
EDA = ROOT / "artifacts/eda"
WORLDS = {"Z": ("Z", EDA / "interface_worlds_v1/Z.pt"), "ZW": ("ZW", EDA / "interface_worlds_zwhite/ZW.pt"),
          "U": ("U", EDA / "interface_worlds_v1/U.pt"), "W": ("W", EDA / "interface_worlds_white/W.pt"),
          "W_s2": ("W", EDA / "interface_worlds_white/W_s2.pt")}


@torch.no_grad()
def run(bundle, heads, pca, arm, encoder, frames, actions, successors, device, batch=16):
    from d4mj.train import autocast_context
    out = {k: [] for k in ("root", "gen", "h", "p_dead", "p_dead_real")}
    for i in range(0, len(frames), batch):
        z, grid = I.encode(encoder, frames[i:i + batch, -4:], device)
        n = len(z)
        s = I.state_of(arm, pca, z, grid).to(device)
        past = actions[i:i + batch, -3:].to(device)
        acts = torch.arange(N, device=device).repeat(n)[:, None]
        with autocast_context(bundle.config):
            state = bundle.world.teacher(s[:, :, None], past).state
            fan = bundle.repeat_state(state, N)
            advanced, features = bundle.advance(fan, acts)
            dead = lambda f: (1 - torch.sigmoid(heads(f)["continuation"][:, -1, 0].float())).view(n, N).cpu()
            out["p_dead"].append(dead(features))
            rz, rgrid = I.encode(encoder, successors[i:i + batch].flatten(0, 1)[:, None], device)
            real = I.state_of(arm, pca, rz, rgrid).to(device)[:, :, None]
            _, real_features = bundle.world.observe_latent(fan, acts, real)
            out["p_dead_real"].append(dead(real_features))
        out["root"].append(s[:, -1].float().cpu())
        out["gen"].append(advanced.latent[:, 0, 0].float().cpu().view(n, N, -1))
        out["h"].append(advanced.history[:, 0].float().cpu().view(n, N, -1))
    return {k: torch.cat(v) for k, v in out.items()}


def main():
    from d4mj.agent import Heads
    from d4mj.experiments import _load_bridge_parent
    sys.path.insert(0, str(LADDER))
    from zwhite import zwhitened_pool
    device = torch.device("cuda")
    pool, _ = zwhitened_pool()                       # installs W and ZW states in interface
    encoder, config = I.load_bridge()
    OUT.mkdir(parents=True, exist_ok=True)
    for block, store in BLOCKS.items():
        rows = [r for f in sorted((EDA / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
        p = torch.stack([r["p_death1"].float() for r in rows])
        keep = [i for i in range(len(rows)) if p[i].max() > p[i].min()]
        rows = [rows[i] for i in keep]
        frames = np.stack([r["frames"][-4:].numpy() for r in rows])
        actions = torch.stack([r["led_to_action"][-3:] for r in rows])
        successors = torch.stack([r["successors"] for r in rows])
        meta = {k: torch.stack([r[k].float() for r in rows]) for k in ("visible", "hidden", "p_death1", "health_delta", "terminated")}
        meta["seed"] = torch.tensor([int(r["seed"]) for r in rows])
        del rows
        torch.save(meta, OUT / f"{block}_meta.pt")
        # canonical H2
        bundle, heads, _ = _load_bridge_parent(I.CHECKPOINT)
        bundle.world.eval(); heads.eval()
        torch.save(run(bundle, heads, pool["pca"], "Z", encoder, frames, actions, successors, device), OUT / f"{block}_H2.pt")
        print(block, "H2", flush=True)
        for name, (arm, path) in WORLDS.items():
            st = torch.load(path, map_location="cpu", weights_only=False)
            b = I.world_bundle(config, encoder, device)
            b.world.load_state_dict(st["world"]); b.world.eval()
            h = Heads(config).to(device); h.load_state_dict(st["heads"]); h.eval()
            torch.save(run(b, h, pool["pca"], arm, encoder, frames, actions, successors, device), OUT / f"{block}_{name}.pt")
            print(block, name, flush=True)


if __name__ == "__main__":
    main()
