"""Blocked moves in imagination, labelled by the visible passability rule (static.py labelled them by the scroll estimator, which
misses 2.4% of visible-rule moved rows, mostly over uniform terrain, and selecting "never-scrolling" roots enriches those).

All futures roots, factual actions, teval's windows. For every future step k whose action is a move, the visible-state passability rule from the true state before the step (onestep.move_table: 0 moved, 1 blocked; lava entries count as moves,
as onestep.classify). On rule-BLOCKED move steps with no earlier false scroll in that rollout:
  false_scroll      the imagined frame k scrolled relative to imagined frame k-1 (scroll.estimate)
  decision logit    the head's move logit (frame gate + target-tile gate) on the imagined window vs the TRUE window
  by target         the blocked target's true tile class / mob
Usage: blocked.py <corrt/corrg world.pt> -> blocked_<name>.json (CPU is enough for the `full` backbone)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
import teval as T  # noqa: E402
from scroll import estimate  # noqa: E402

TARGET = torch.tensor([31, 30, 32, 22, 40])
DR = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    from onestep import move_table
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    path = Path(sys.argv[1])
    world, st = T.load_world(path, device)
    cache = T.build_cache(st["args"]["pool"], device)
    R = len(meta["seed"])
    states = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0, :15]], 1).float()   # before step k
    cats = torch.stack([move_table(states[:, k])[0] for k in range(16)], 1)                             # [R,16,17]
    tiles = states[..., :1071].reshape(R, 16, 7, 9, 17).argmax(-1)
    lava = torch.zeros(R, 16, 17, dtype=torch.bool)
    for a, (dr, dc) in DR.items():
        lava[..., a] = (tiles[:, :, 3 + dr, 4 + dc] == 14) & (states[..., 1520] < 0.5)
    fa, ca = cache["fut_a"], cache["ctx_a"]
    act_cat = cats.gather(2, fa[..., None])[..., 0]
    is_move = (fa >= 1) & (fa <= 4)
    blocked = is_move & (act_cat == 1) & ~lava.gather(2, fa[..., None])[..., 0]
    imag_logit, true_logit = torch.zeros(R, 16), torch.zeros(R, 16)
    false_scroll = torch.zeros(R, 16)
    for i in range(0, R, 16):
        ctx, fut = cache["ctx"][i:i + 16].float(), cache["fut"][i:i + 16].float()
        b = len(ctx)
        imag, true = [ctx[:, j] for j in range(4)], [ctx[:, j] for j in range(4)]
        acts = [ca[i:i + b, j] for j in range(3)]
        for k in range(16):
            w_ = 4 if k == 0 else 5
            a = torch.stack(acts[-(w_ - 1):] + [fa[i:i + b, k]], 1).to(device)
            tgt = TARGET[fa[i:i + b, k].clamp(max=4)].to(device)
            ar = torch.arange(b, device=device)
            for hist, store in ((imag, imag_logit), (true, true_logit)):
                with autocast_context(config):
                    h, ha = world.backbone_full(torch.stack(hist[-w_:], 1).to(device).float(), a)
                    logit = world.frame(ha[:, -1]).float()[:, 0]
                    if hasattr(world, "target_gate"):
                        logit = logit + world.target_gate(h[ar, -1, tgt]).float()[:, 0]
                store[i:i + b, k] = logit.cpu()
            g = T.step(world, torch.stack(imag[-w_:], 1), a.cpu(), device, config)
            false_scroll[i:i + b, k] = (estimate(imag[-1], g) != 0).float()
            imag.append(g); true.append(fut[:, k]); acts.append(fa[i:i + b, k])
    before = (false_scroll.cumsum(1) - false_scroll) == 0
    sel = blocked & before
    rv_mobs = states[..., 1071:1512].reshape(R, 16, 7, 9, 7)
    by = {}
    for r, k in zip(*torch.where(sel)):
        a = int(fa[r, k]); tr, tc = 3 + DR[a][0], 4 + DR[a][1]
        mob = [n for n, c in (("zombie", 0), ("cow", 1), ("skeleton", 2)) if rv_mobs[r, k, tr, tc, c] > 0]
        key = "mob:" + mob[0] if mob else f"tile:{int(tiles[r, k, tr, tc])}"
        d = by.setdefault(key, [0, 0, 0])
        d[0] += 1; d[1] += int(false_scroll[r, k] > 0); d[2] += int(true_logit[r, k] > 0)
    out = {"blocked_first_steps": int(sel.sum()), "all_blocked_steps": int(blocked.sum()),
           "false_scroll_rate": float(false_scroll[sel].mean()),
           "by_depth": {k: float(false_scroll[:, k - 1][sel[:, k - 1]].mean()) for k in (1, 2, 4, 8, 16) if sel[:, k - 1].any()},
           "decision_true_window_mean": float(true_logit[sel].mean()),
           "decision_imagined_window_mean": float(imag_logit[sel].mean()),
           "share_positive_true_window": float((true_logit[sel] > 0).float().mean()),
           "share_positive_imagined_window": float((imag_logit[sel] > 0).float().mean()),
           "on_false_scrolls_true_vs_imagined": [float(true_logit[sel & (false_scroll > 0)].mean()),
                                                 float(imag_logit[sel & (false_scroll > 0)].mean())] if (sel & (false_scroll > 0)).any() else None,
           "by_target_n_falsescroll_truepositive": {k: v for k, v in sorted(by.items(), key=lambda kv: -kv[1][0])}}
    (HERE / f"blocked_{st['name']}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
