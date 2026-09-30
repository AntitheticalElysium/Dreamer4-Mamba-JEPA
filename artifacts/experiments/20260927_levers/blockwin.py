"""E5k. Why corrt-18k misjudges blocked moves as scrolls on TRUE inputs with a 5-frame window but not with 4.

Factual futures, steps k >= 1 whose factual action is a move that the visible passability rule (choices.move_table
on the true state at step k, lava entries excluded as in onestep.classify) labels blocked. For each such step the
head's move-decision logit (frame logit + target-tile gate for corrt; frame logit alone for corrg) is read on TRUE
frames ending at the current frame, under four inputs:
  w4        the last 4 frames and their outgoing actions (current frame at time embedding 3)
  w5        the last 5 frames (current frame at time embedding 4): teval's rollout window
  w5dup     w5 with its oldest frame and action replaced by copies of the second-oldest: the content of w4, the
            positions of w5
  w5noop    w5 with only the oldest action replaced by NOOP (0)
A positive logit means the head scrolls (it treats the move as successful). Split by the target tile and by whether
the previous action was DO. Usage: blockwin.py <world.pt> -> blockwin_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
import teval as T  # noqa: E402

TARGET = torch.tensor([31, 30, 32, 22, 40])
DR = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
TILES = {2: "grass", 3: "water", 4: "stone", 5: "tree", 7: "path", 8: "coal", 9: "iron", 10: "diamond",
         11: "table", 12: "furnace", 13: "sand", 15: "plant", 16: "ripe_plant"}


@torch.no_grad()
def main():
    from choices import move_table
    from d4mj.config import config_from_dict
    from d4mj.train import autocast_context
    import spatial as S
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    world, st = T.load_world(Path(sys.argv[1]), device)
    cache = T.build_cache(st["args"]["pool"], device)
    R = len(meta["seed"])
    states = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0, :15]], 1).float()   # state at step k
    cats = torch.stack([move_table(states[:, k])[0] for k in range(16)], 1)                          # [R,16,17]
    tiles = states[..., :1071].reshape(R, 16, 7, 9, 17).argmax(-1)
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    frames = torch.cat([cache["ctx"].float(), cache["fut"].float()], 1)                                 # [R,20]: t-3..t, then futures
    acts = torch.cat([cache["ctx_a"], cache["fut_a"]], 1)                                               # [R,19]
    rows = []
    for k in range(1, 16):
        a = cache["fut_a"][:, k]
        m = (a >= 1) & (a <= 4) & alive[:, k]
        m &= cats[:, k].gather(1, a[:, None].clamp(max=16))[:, 0] == 1
        for r in torch.where(m)[0].tolist():
            ai = int(a[r]); tr, tc = 3 + DR[ai][0], 4 + DR[ai][1]
            rows.append((r, k, ai, int(tiles[r, k, tr, tc]), int(acts[r, 3 + k - 1])))
    cur = lambda r, k: 3 + k                                          # index of the current frame in `frames`

    def window(r, k, variant):
        c = cur(r, k)
        if variant == "w4":
            return frames[r, c - 3:c + 1], acts[r, c - 3:c + 1]
        f, x = frames[r, c - 4:c + 1].clone(), acts[r, c - 4:c + 1].clone()
        if variant == "w5dup":
            f[0], x[0] = f[1], x[1]
        elif variant == "w5noop":
            x[0] = 0
        return f, x

    logits = {}
    for variant in ("w4", "w5", "w5dup", "w5noop"):
        out = []
        for s in range(0, len(rows), 128):
            chunk = rows[s:s + 128]
            fw, aw = zip(*[window(r, k, variant) for r, k, _, _, _ in chunk])
            fw, aw = torch.stack(fw).to(device), torch.stack(aw).to(device)
            with autocast_context(config):
                h, ha = world.backbone_full(fw, aw)
                logit = world.frame(ha[:, -1]).float()[:, 0]
                if hasattr(world, "target_gate"):
                    tgt = TARGET[torch.tensor([x[2] for x in chunk])].to(device)
                    logit = logit + world.target_gate(h[torch.arange(len(chunk), device=device), -1, tgt]).float()[:, 0]
            out.append(logit.cpu())
        logits[variant] = torch.cat(out)
    kinds = torch.tensor([x[3] for x in rows]); prev_do = torch.tensor([x[4] == 5 for x in rows])
    res = {"world": st["name"], "blocked_steps": len(rows), "variants": {}}
    for v, l in logits.items():
        pos = l > 0
        row = {"positive_rate": float(pos.float().mean()), "mean_logit": float(l.mean()),
               "positive_if_prev_DO": float(pos[prev_do].float().mean()),
               "positive_if_not_prev_DO": float(pos[~prev_do].float().mean()), "by_target": {}}
        for kv, name in TILES.items():
            m = kinds == kv
            if int(m.sum()) >= 10:
                row["by_target"][name] = {"n": int(m.sum()), "positive": float(pos[m].float().mean()),
                                          "positive_if_prev_DO": float(pos[m & prev_do].float().mean()) if (m & prev_do).any() else None}
        res["variants"][v] = row
    res["prev_DO_share"] = float(prev_do.float().mean())
    res["rows"] = [{"root": r, "step": k, "action": a, "target_tile": t, "prev_action": p} for r, k, a, t, p in rows]
    (HERE / f"blockwin_{st['name']}.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({v: {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if k != "by_target"}
                      for v, r in res["variants"].items()}), flush=True)


if __name__ == "__main__":
    main()
