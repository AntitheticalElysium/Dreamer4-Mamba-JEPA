"""E5k. Teacher-forced one-step quality of a six-frame per-tile world as a function of the time position of the
current frame, on TRUE frames.

Factual futures, steps k >= 1 (the current frame is future frame k-1 or the root; at least 5 true frames exist).
The next frame is predicted from the last w true frames and actions, w = 1..5, so the current frame sits at learned
time embedding w-1. `w4at4` places the w=4 window at positions 1..4 by prepending a copy of its oldest frame and
action: the content of w=4 with the current frame at position 4, which separates position from context content.
Error is the squared error summed over tokens / V (teval's normalizer), alive steps,
all 1,002 roots, per class (onestep.classify-style visible rule on the true state at step k: moved / blocked move /
other). Usage: posprofile.py <world.pt> -> posprofile_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
import teval as T  # noqa: E402


@torch.no_grad()
def main():
    from choices import move_table
    from d4mj.config import config_from_dict
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    world, st = T.load_world(Path(sys.argv[1]), device)
    cache = T.build_cache(st["args"]["pool"], device)
    R = len(meta["seed"])
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    frames = torch.cat([cache["ctx"].float(), cache["fut"].float()], 1)                 # [R,20]
    acts = torch.cat([cache["ctx_a"], cache["fut_a"]], 1)                               # [R,19]
    fut = cache["fut"].float()
    V = float(((fut - fut.flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    states = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0, :15]], 1).float()
    cats = torch.stack([move_table(states[:, k])[0] for k in range(16)], 1)            # [R,16,17]
    variants = {f"w{w}": w for w in range(1, 6)} | {"w4at4": 4}
    res = {"world": st["name"], "V": V, "variants": {}}
    for name, w in variants.items():
        err, cls_all = [], []
        for k in range(1, 16):
            c = 3 + k                                                                   # current frame index
            f, a = frames[:, c - w + 1:c + 1], acts[:, c - w + 1:c + 1]
            if name == "w4at4":
                f, a = torch.cat([f[:, :1], f], 1), torch.cat([a[:, :1], a], 1)
            pred = torch.cat([T.step(world, f[i:i + 32], a[i:i + 32], device, config) for i in range(0, R, 32)])
            e = ((pred - fut[:, k]) ** 2).sum((-1, -2))
            fa = cache["fut_a"][:, k]
            mv = (fa >= 1) & (fa <= 4)
            cat = cats[:, k].gather(1, fa[:, None])[:, 0]
            cl = torch.where(mv & (cat == 0), 0, torch.where(mv & (cat == 1), 1, 2))
            m = alive[:, k]
            err.append(e[m]); cls_all.append(cl[m])
        e, cl = torch.cat(err), torch.cat(cls_all)
        res["variants"][name] = {"all_over_V": float(e.mean() / V), "n": len(e)} | {
            lab: float(e[cl == i].mean() / V) for i, lab in enumerate(("moved", "blocked", "other"))}
    (HERE / f"posprofile_{st['name']}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(st["name"], json.dumps({k: {x: round(y, 4) for x, y in v.items() if x != "n"} for k, v in res["variants"].items()}), flush=True)


if __name__ == "__main__":
    main()
