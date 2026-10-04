"""Stage 2 prerequisite. The view scroll between two consecutive frames, estimated from their tokens alone.

Craftax scrolls the 7 x 9 map view by exactly one tile when a move succeeds. For two frames' tokens a, b [.., 81, D]
(layer-normed, map = rows 0..6 of the 9 x 9 grid) the estimate is the shift s in {none, up, down, left, right}
minimizing the mean squared token difference over the map cells both frames share: b[r, c] ~ a[r + dr, c + dc]
(the corr head's NEIGHBOURS convention). Parameter-free, no gradient; the same rule reads observed and imagined
frames. main(): its accuracy on the futures roots (teval cache, all 17 actions, key 0) against the simulator's own
moved / blocked label (onestep.classify, lava entries counted as moves).
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))                 # none + tworld.NEIGHBOURS
MOVE_SHIFT = {1: 3, 2: 4, 3: 1, 4: 2}                               # action (left, right, up, down) -> SHIFTS index


def estimate(a, b):
    """a, b [..., 81, D] -> [...] index into SHIFTS."""
    ga, gb = a[..., :63, :].unflatten(-2, (7, 9)).float(), b[..., :63, :].unflatten(-2, (7, 9)).float()
    errs = []
    for dr, dc in SHIFTS:
        rs, cs = slice(max(0, -dr), 7 - max(0, dr)), slice(max(0, -dc), 9 - max(0, dc))
        rt, ct = slice(max(0, dr), 7 + min(0, dr)), slice(max(0, dc), 9 + min(0, dc))
        errs.append((gb[..., rs, cs, :] - ga[..., rt, ct, :]).square().mean((-1, -2, -3)))
    return torch.stack(errs, -1).argmin(-1)


def main():
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    import teval as T
    from onestep import classify
    meta, _, _ = T.split()
    cache = T.build_cache("raw", torch.device("cuda"))
    cls, _ = classify(meta)                                          # [R,17]: 0 moved, 1 blocked
    root, one = cache["ctx"][:, -1].float(), cache["one"].float()
    est = torch.cat([estimate(root[i:i + 32, None], one[i:i + 32]) for i in range(0, len(root), 32)])     # [R,17]
    truth = torch.zeros_like(est)
    for a, s in MOVE_SHIFT.items():
        truth[:, a] = torch.where(cls[:, a] == 0, s, 0)
    ok = est == truth
    out = {"accuracy_all": float(ok.float().mean()),
           "accuracy_moved": float(ok[cls == 0].float().mean()), "n_moved": int((cls == 0).sum()),
           "accuracy_blocked": float(ok[cls == 1].float().mean()), "n_blocked": int((cls == 1).sum()),
           "accuracy_non_move": float(ok[cls >= 2].float().mean()),
           "moved_detected_as_none": float((est[cls == 0] == 0).float().mean()),
           "non_moved_detected_as_scroll": float((est[cls != 0] != 0).float().mean())}
    # factual 16-step futures: consecutive true frames, scroll vs the action taken and the next frame's class
    fut = torch.cat([cache["ctx"][:, -1:].float(), cache["fut"].float()], 1)                               # [R,17,..]
    est_f = torch.cat([estimate(fut[i:i + 16, :-1], fut[i:i + 16, 1:]) for i in range(0, len(fut), 16)])  # [R,16]
    out["futures_scroll_rate"] = float((est_f != 0).float().mean())
    out["futures_scroll_on_move_actions"] = float((est_f != 0)[(cache["fut_a"] >= 1) & (cache["fut_a"] <= 4)].float().mean())
    out["futures_scroll_on_other_actions"] = float((est_f != 0)[(cache["fut_a"] == 0) | (cache["fut_a"] > 4)].float().mean())
    agree = est_f == torch.tensor([0, 3, 4, 1, 2])[cache["fut_a"].clamp(max=4)]
    out["futures_scroll_direction_matches_action"] = float(agree[est_f != 0].float().mean())
    (HERE / "scroll.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
