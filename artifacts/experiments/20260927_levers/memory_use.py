"""E1c. What does the Mamba's memory help predict? Memory benefit by transition type, E1/E1b worlds.

Held-out DEV/FINAL windows of 128 frames (256 windows, seed 3), encoded by the frozen Raw encoder (eval mode, as
the cache). For every predicted position t >= 16 (long context), each world's teacher-forced error with its full
history vs le-wm's 3-frame window; benefit = 1 - err(full) / err(window3), per transition type:
  moved / blocked      move actions, pixel scroll classifier (98.8% accurate when it decides)
  exact_revisit        the next frame's map area (rows 0-48) is pixel-identical to a frame 4+ steps back and to none
                       of the last 3 -- recall of content that left the 3-frame window
  sleep_onset / wake   the sleep render flag (sleep.asleep) switches on / off
  other                everything else
Worlds: whatever E1 / E1b saved (L4, L16, L64, L4to16, L4to64, L16x3).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DIAG = ROOT / "artifacts/experiments/20260926_diagnosis"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(DIAG))
import context_length as C  # noqa: E402

T_WIN, N_WIN = 128, 256


def windows():
    from insample import STORES
    index = []
    for store in STORES:
        for s in sorted(store.glob("shard-*.pt")):
            for j, e in enumerate(torch.load(s, weights_only=False, mmap=True)["episodes"]):
                if e["split"] in ("dev", "final") and len(e["actions_taken"]) >= T_WIN:
                    index.append((s, j, len(e["actions_taken"])))
    g = torch.Generator().manual_seed(3)
    w = torch.tensor([n - T_WIN + 2 for _, _, n in index], dtype=torch.float64)
    picks = torch.multinomial(w, N_WIN, replacement=True, generator=g).tolist()
    items = [(index[p][0], index[p][1], int(torch.randint(index[p][2] - T_WIN + 2, (), generator=g))) for p in picks]
    frames, actions = [None] * N_WIN, [None] * N_WIN
    by = {}
    for k, (s, j, t) in enumerate(items):
        by.setdefault(s, []).append((k, j, t))
    for s, rows in by.items():
        eps = torch.load(s, weights_only=False, mmap=True)["episodes"]
        for k, j, t in rows:
            frames[k] = torch.as_tensor(np.asarray(eps[j]["observations"][t:t + T_WIN]))
            actions[k] = eps[j]["actions_taken"][t:t + T_WIN - 1].clone()
    return torch.stack(frames), torch.stack(actions)


def types(frames, actions):
    from coverage import classify
    from sleep import asleep
    n, T = actions.shape
    out = np.full((n, T), "other", dtype=object)
    before, after, a = frames[:, :-1].flatten(0, 1), frames[:, 1:].flatten(0, 1), actions.flatten()
    move = (a >= 1) & (a <= 4)
    sc = torch.zeros(len(a), dtype=torch.long)
    sc[move] = classify(before[move], after[move], a[move])
    sl_b, sl_a = asleep(before), asleep(after)
    flat = out.reshape(-1)
    flat[(move & (sc == 1)).numpy()] = "moved"
    flat[(move & (sc == -1)).numpy()] = "blocked"
    flat[(sl_a & ~sl_b).numpy()] = "sleep_onset"
    flat[(~sl_a & sl_b).numpy()] = "wake"
    # exact-match test via a random projection hash of each frame's map area (equal frames -> equal hash;
    # different frames collide with probability ~0 at float64)
    r = torch.randn(49 * 63 * 3, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    h = frames[:, :, :49].flatten(2).double() @ r                                    # [n, T+1]
    for i in range(n):
        for t in range(4, T):
            nxt = h[i, t + 1]
            recent = (h[i, t - 2:t + 1] == nxt).any()
            old = (h[i, :t - 2] == nxt).any()
            if bool(old) and not bool(recent) and out[i, t] in ("moved", "other", "blocked"):
                out[i, t] = "exact_revisit"
    return out


@torch.no_grad()
def main():
    device = torch.device("cuda")
    from d4mj.checkpoint import read_lewm_bundle
    from d4mj.config import config_from_dict
    from d4mj.world_api import ModelBundle
    from d4mj.train import autocast_context
    payload = read_lewm_bundle(ROOT / "artifacts/lewm_m4_canonical/raw/joint/step-010000.pt")
    config = config_from_dict(payload["config"])
    bundle = ModelBundle.create(config)
    bundle.encoder.load_state_dict(payload["modules"]["encoder"])
    enc = bundle.encoder.to(device).freeze()
    frames, actions = windows()
    kinds = types(frames, actions)                                                   # [n, T-1]
    z = torch.stack([torch.cat([enc(frames[i, j:j + 64][None].to(device)).float()[0] for j in range(0, T_WIN, 64)])
                     for i in range(N_WIN)])                                          # [n, T, 1, D]
    result = {"counts": {k: int((kinds[:, 15:] == k).sum()) for k in np.unique(kinds)}}
    for name in ("L4", "L16", "L64", "L4to16", "L4to64", "L16x3", "L64b24", "L4b512", "L16b100", "L4to64b24"):
        path = C.OUT / f"{name}.pt"
        if not path.exists():
            continue
        world = C.new_world(config, device)
        world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"]); world.eval()
        full, w3 = [], []
        with autocast_context(config):
            for i in range(0, N_WIN, 16):
                zi, ai = z[i:i + 16].to(device), actions[i:i + 16].to(device)
                full.append(((world.teacher(zi, ai).predicted.float() - zi[:, 1:].float()) ** 2).mean(-1)[:, :, 0].cpu())
                w3.append(((C.window3(world, zi, ai).float() - zi[:, 1:].float()) ** 2).mean(-1)[:, :, 0].cpu())
        full, w3 = torch.cat(full)[:, 15:], torch.cat(w3)[:, 15:]
        k = kinds[:, 15:]
        row = {}
        for kind in np.unique(k):
            m = torch.from_numpy(k == kind)
            row[kind] = {"benefit": float(1 - full[m].mean() / w3[m].mean()), "full": float(full[m].mean()),
                         "window3": float(w3[m].mean())}
        row["all"] = {"benefit": float(1 - full.mean() / w3.mean())}
        result[name] = row
        print(name, json.dumps({kk: round(v["benefit"], 3) for kk, v in row.items()}), flush=True)
    (HERE / "memory_use.json").write_text(json.dumps(result, indent=2) + "\n")
    print("counts", result["counts"])


if __name__ == "__main__":
    main()
