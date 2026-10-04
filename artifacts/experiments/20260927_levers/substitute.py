"""Ground-truth substitution in a per-tile world's 16-step imagination: which component's errors cause the rest.

Futures roots, factual actions, teval's windows (4 frames, then the last 5). After every imagined step, one component
of the imagined frame is replaced by the TRUE tokens before it is fed back:
  none        teval's imagination
  entering    the row / column the TRUE transition scrolled in (scroll.estimate on consecutive true frames;
              nothing when it did not scroll)
  hud         the 18 HUD tokens
  player      the player tile (31)
  all3        entering + hud + player
Read-out: imagined squared error / the group's own variance at depths 1, 4, 8, 16 for the groups of where.py
(player, near, interior, edge, hud), alive roots. Error is measured BEFORE the replacement, so a substituted group
reads its one-step error given a history in which it was right; the question is what happens to the OTHER groups. Usage: substitute.py <world.pt> -> substitute_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
import teval as T  # noqa: E402
from scroll import estimate  # noqa: E402
from where import GROUPS  # noqa: E402

ENTERING = {1: list(range(9)), 2: list(range(54, 63)), 3: [r * 9 for r in range(7)], 4: [r * 9 + 8 for r in range(7)]}
ARMS = {"none": (), "entering": ("entering",), "hud": ("hud",), "player": ("player",),
        "all3": ("entering", "hud", "player")}


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    path = Path(sys.argv[1])
    world, st = T.load_world(path, device)
    cache = T.build_cache(st["args"]["pool"], device)
    R = len(meta["seed"])
    fut, ca, fa = cache["fut"], cache["ctx_a"], cache["fut_a"]
    var = ((fut.float() - fut.float().flatten(0, 1).mean(0)) ** 2).sum(-1).mean((0, 1))              # [81]
    out = {}
    for arm, parts in ARMS.items():
        gerr = torch.empty(R, T.H, 81)
        for i in range(0, R, 16):
            ctx, true_f = cache["ctx"][i:i + 16].float(), fut[i:i + 16].float()
            b = len(ctx)
            frames, hist, prev = [ctx[:, j] for j in range(4)], [ca[i:i + b, j] for j in range(3)], ctx[:, -1]
            for k in range(T.H):
                w_ = 4 if k == 0 else 5
                a = torch.stack(hist[-(w_ - 1):] + [fa[i:i + b, k]], 1)
                g = T.step(world, torch.stack(frames[-w_:], 1), a, device, config)
                true = true_f[:, k]
                gerr[i:i + b, k] = ((g - true) ** 2).sum(-1)
                if "entering" in parts:
                    shift = estimate(prev, true)
                    for s, cells in ENTERING.items():
                        sel = shift == s
                        if sel.any():
                            rows, cols = torch.where(sel)[0][:, None], torch.tensor(cells)[None]
                            g[rows, cols] = true[rows, cols]
                if "hud" in parts:
                    g[:, 63:] = true[:, 63:]
                if "player" in parts:
                    g[:, 31] = true[:, 31]
                frames.append(g); hist.append(fa[i:i + b, k]); prev = true
        out[arm] = {}
        for k in (1, 4, 8, 16):
            m = alive[:, k - 1]
            e = gerr[m, k - 1]
            out[arm][k] = {"total_over_V": float(e.sum(-1).mean() / var.sum())} | \
                          {g: float(e[:, tok].sum(-1).mean() / var[tok].sum()) for g, tok in GROUPS.items()}
        print(arm, json.dumps({k: {g: round(v, 3) for g, v in r.items()} for k, r in out[arm].items()}), flush=True)
    (HERE / f"substitute_{st['name']}.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
