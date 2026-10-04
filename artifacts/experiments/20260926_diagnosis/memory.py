"""D22. Stuffed Mamba's signature in our worlds: does the recurrent state keep accumulating past the training
length (no learned forgetting), where the one-step error degrades (D21)?

Chen et al. 2025 (arXiv 2410.07145): Mamba trained on contexts too short for its state size does not learn to
forget; its state keeps growing past the training length and outputs degrade. Same 256 x 96-frame TRAIN windows
as D21 (seed 2). For each world, the teacher's recurrent state after consuming t frames, t in {2,4,6,8,16,32,
64,96}: mean over layers of the SSM state's RMS, relative to t = 4; and the RMS of the history (the output the
projector reads).
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import bnmode  # noqa: E402

POSITIONS = (2, 4, 6, 8, 16, 32, 64, 96)


@torch.no_grad()
def main():
    import interface as I
    from whiten import whitened_pool
    device = torch.device("cuda")
    bnmode.BATCHES, bnmode.B, bnmode.T = 8, 32, 96
    frames, actions = bnmode.windows(2)
    worlds = {"joint": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt", "joint"),
              "bridge": bnmode.load(ROOT / "artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt", "bridge")}
    enc = worlds["joint"].encoder.to(device).eval()
    pool, _ = whitened_pool()
    _, config = I.load_bridge()
    for name, path in (("Z", "interface_worlds_v1/Z.pt"), ("U", "interface_worlds_v1/U.pt")):
        b = I.world_bundle(config, enc, device)
        b.world.load_state_dict(torch.load(ROOT / "artifacts/eda" / path, map_location="cpu", weights_only=False)["world"])
        worlds[name] = b
    acc = {w: {t: [[], []] for t in POSITIONS} for w in worlds}
    for i in range(0, len(frames), 32):
        f, a = frames[i:i + 32].to(device), actions[i:i + 32].to(device)
        parts = [enc.export(f[j:j + 4], grid=4) for j in range(0, len(f), 4)]
        z = torch.cat([p[0] for p in parts]).float()
        grid = torch.cat([p[2] for p in parts]).float().flatten(2)
        u = I.state_of("U", pool["pca"], z[:, :, 0].cpu(), grid.cpu()).to(device)[:, :, None]
        for w, b in worlds.items():
            s = u if w == "U" else z
            world = b.world.to(device).eval()
            for t in POSITIONS:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    st = world.teacher(s[:, :t], a[:, :t - 1]).state
                ssm = torch.stack([m.ssm.float().pow(2).mean() for m in st.memory]).mean().sqrt()
                acc[w][t][0].append(float(ssm)); acc[w][t][1].append(float(st.history.float().pow(2).mean().sqrt()))
    result = {}
    for w in worlds:
        base = sum(acc[w][4][0]) / len(acc[w][4][0])
        result[w] = {str(t): {"ssm_rms_rel_t4": sum(v[0]) / len(v[0]) / base, "history_rms": sum(v[1]) / len(v[1])}
                     for t, v in acc[w].items()}
        print(w, json.dumps({t: round(r["ssm_rms_rel_t4"], 2) for t, r in result[w].items()}),
              "history", json.dumps({t: round(r["history_rms"], 2) for t, r in result[w].items()}), flush=True)
    (HERE / "memory.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
