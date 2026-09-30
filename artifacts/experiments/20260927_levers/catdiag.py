"""E2 diagnosis. Where exactly the categorical per-tile world loses, in its own currency (codes) and in the shared one.

Futures roots (teval cache, Raw tokens), codebook raw_K4096. Per map cell (63 = 7 x 9), one step, all 17 actions,
key 0, the true successor's code c* = q(true token) against the root's code c0 = q(root token):
  stay         c* == c0
  flip         c* != c0 while the cell's simulator content (tile class and the 7 mob channels) is unchanged
               -- a contextual code flip: the token moved across a Voronoi boundary without anything happening there
  content      c* != c0 and the cell's content changed (scroll, placed/mined tile, mob moved)
For each world: code accuracy argmax == c* on each of the three, and copy's (c0 == c*: 1 / 0 / 0).
Worlds: the categorical world (teacher-forced CE, logits argmax); continuous worlds read through the same codebook
(q(prediction)) -- the continuous teacher-forced control (residual_raw_teacher_s7), residual and corrt (suffix).
Floors, independent of any world:
  snap floor   ||q(x) - x||^2 of the TRUE next frame, / copy error by transition class, / V by depth: the least
               error any code-output world scores on teval's continuous metric
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260921_readout_ladder"))
import teval as T  # noqa: E402
from tworld import quantize  # noqa: E402

W = ROOT / "artifacts/eda/levers_tworlds_v1"
WORLDS = ["categorical_raw_teacher_s7_K4096", "residual_raw_teacher_s7", "residual_raw_suffix_s7", "corrt_raw_suffix_s7"]
CB = ROOT / "artifacts/eda/levers_codebooks_v1/raw_K4096.pt"
MAP = torch.tensor(T.MAP)


def content(v):
    """[..., 1534] visible state -> [..., 63] (tile class, mob channels) as one comparable tensor."""
    tiles = v[..., :1071].reshape(*v.shape[:-1], 63, 17).argmax(-1, keepdim=True).float()
    mobs = v[..., 1071:1512].reshape(*v.shape[:-1], 63, 7)
    return torch.cat([tiles, mobs], -1)


@torch.no_grad()
def main():
    from onestep import CLASSES, classify
    device = torch.device("cuda")
    meta, _, _ = T.split()
    cache = T.build_cache("raw", device)
    codes = torch.load(CB, weights_only=False)["codes"].float().to(device)
    R = len(meta["seed"])
    cls, _ = classify(meta)
    root, one, fut = cache["ctx"][:, -1].float(), cache["one"].float(), cache["fut"].float()
    c0 = torch.cat([quantize(root[i:i + 64].to(device), codes).cpu() for i in range(0, R, 64)])              # [R,81]
    cstar = torch.cat([quantize(one[i:i + 16].to(device), codes).cpu() for i in range(0, R, 16)])            # [R,17,81]
    same = content(meta["onestep_visible"][:, 0]) == content(meta["root_visible"])[:, None]
    unchanged = same.all(-1)                                                                                  # [R,17,63]
    stay = (cstar[..., MAP] == c0[:, None, MAP])
    groups = {"stay": stay, "flip": ~stay & unchanged, "content": ~stay & ~unchanged}
    out = {"cells": {g: int(m.sum()) for g, m in groups.items()}}
    out["flip_share_of_changed_codes"] = float(groups["flip"].sum() / (~stay).sum())
    # floors
    snap_one = torch.cat([codes[cstar[i:i + 16].to(device)].cpu() for i in range(0, R, 16)])
    floor = ((snap_one - one) ** 2).sum((-1, -2))
    cp = ((root[:, None] - one) ** 2).sum((-1, -2))
    out["snap_floor_x_copy"] = {c: float(floor[cls == i].mean() / cp[cls == i].mean()) for i, c in enumerate(CLASSES)}
    out["snap_floor_x_copy"]["all"] = float(floor.mean() / cp.mean())
    V = float(((fut - fut.flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    snapf = lambda k: torch.cat([codes[quantize(fut[i:i + 64, k].to(device), codes)].cpu() for i in range(0, R, 64)])
    out["snap_floor_over_V"] = {k: float(((snapf(k - 1) - fut[:, k - 1]) ** 2).sum((-1, -2))[alive[:, k - 1]].mean() / V)
                                for k in (1, 4, 8, 16)}
    out["copy_over_V"] = {k: float(((root - fut[:, k - 1]) ** 2).sum((-1, -2))[alive[:, k - 1]].mean() / V)
                          for k in (1, 4, 8, 16)}
    print(json.dumps(out), flush=True)
    from d4mj.config import config_from_dict
    import spatial as S
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    ctx, ca = cache["ctx"], cache["ctx_a"]
    for name in WORLDS:
        world, _ = T.load_world(W / f"{name}.pt", device)
        pred = torch.empty(R, T.N, 81, dtype=torch.long)
        for i in range(0, R, 4):                        # 4 roots x 17 actions: the categorical logits are 4 x 81 x 4096 per row
            b = min(4, R - i)
            fan = ctx[i:i + b].float().repeat_interleave(T.N, 0)
            acts = torch.cat([ca[i:i + b].repeat_interleave(T.N, 0), torch.arange(T.N).repeat(b)[:, None]], 1)
            nxt = T.step(world, fan, acts, device, config)                                  # [b*17,81,192]
            pred[i:i + b] = quantize(nxt.to(device), codes).cpu().view(b, T.N, 81)
        ok = pred[..., MAP] == cstar[..., MAP]
        row = {g: float(ok[m].float().mean()) for g, m in groups.items()}
        row["all"] = float(ok.float().mean())
        moved = (cls == 0)[:, :, None].expand_as(ok)
        row["content_on_moved"] = float(ok[groups["content"] & moved].float().mean())
        row["content_not_moved"] = float(ok[groups["content"] & ~moved].float().mean())
        out[name] = row
        print(name, json.dumps(row), flush=True)
        del world
        torch.cuda.empty_cache()
    (HERE / "catdiag.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
