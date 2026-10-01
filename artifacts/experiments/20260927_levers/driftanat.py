"""E11b. Anatomy of the deterministic drift that E11 found (stochdiag: ~90% of depth-16 error is the model's own, ~80% of its
growth on static map cells). Is it the view POSITION (a wrong or missed scroll shifts the whole map) or the CONTENT?

Same data and rollouts as stochdiag.py (diagnosis futures, five sampled futures per root, imagined rollout of the shared
16 actions, teval's window convention). Per root and depth k:
  offsets    the camera offset accumulated from scroll.estimate between consecutive frames (0.994 accurate on true
             frames): imagined O^_k from the imagined frames (root -> g_1 -> ... -> g_k), true O^s_k for each sample s
  aligned    O^_k == O^0_k (sample 0, the factual future)
  agree      the five samples' true offsets all equal (position is not random at this depth)
Reported, on depths where all five samples are alive and only where `agree` (so position error is the model's alone):
  wrong_rate          share of (root, depth) with the imagined offset wrong, by depth
  first_wrong         depth of the first wrong offset per root (histogram); the step that caused it: the true action's
                      class (moved / blocked / other: onestep.classify-style visible rule on sample 0) and the imagined vs
                      true scroll at that step
  excess split        stochdiag's excess (|g - mu|^2 - s^2/5, / V), static + entering + mob + hud + player, split by aligned
                      vs wrong, and each group's share of all depth-16 excess
  realigned           for wrong cases, the imagined map shifted by (O^0 - O^) and compared with mu on the cells both cover:
                      the static excess that remains once position is corrected = content drift
Readings, declared before running:
  position_dominated  if wrong cases carry >= 50% of the depth-16 excess on map cells, AND realigning removes >= 50% of
                      their map-cell excess
  content_dominated   if aligned cases carry >= 50% of the depth-16 map-cell excess
Usage: driftanat.py <world.pt> ... -> evals/driftanat_<name>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import stochdiag as SD  # noqa: E402
T = SD.T
H, S = SD.H, SD.S


def offsets(shift_idx):
    """[..., k] scroll indices -> [..., k, 2] cumulative camera offsets."""
    from scroll import SHIFTS
    return torch.tensor(SHIFTS)[shift_idx].cumsum(-2)


def shift_map(x, dr, dc):
    """Map tokens [81, D] shifted so cell (r, c) takes x[r + dr, c + dc]; returns shifted map [63, D] and validity [63]."""
    m = x[:63].view(7, 9, -1)
    out = torch.zeros_like(m); valid = torch.zeros(7, 9, dtype=torch.bool)
    rs, re_ = max(0, -dr), 7 - max(0, dr)
    cs, ce = max(0, -dc), 9 - max(0, dc)
    if rs < re_ and cs < ce:
        out[rs:re_, cs:ce] = m[rs + dr:re_ + dr, cs + dc:ce + dc]
        valid[rs:re_, cs:ce] = True
    return out.view(63, -1), valid.view(63)


@torch.no_grad()
def main():
    from d4mj.config import config_from_dict
    from onestep import classify  # noqa: F401  (onestep's visible rule, as blockwin)
    from choices import move_table
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    V = float(((cache["fut"].float() - cache["fut"].float().flatten(0, 1).mean(0)) ** 2).sum((-1, -2)).mean())
    cls, _ = SD.classes(meta, fut5[:, 0], root)
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)                          # [R,16]
    # true offsets per sample
    true_shift = []
    for s in range(S):                                                             # one sample at a time (RAM)
        prev = torch.cat([root[:, None], fut5[:, s, :-1]], 1)
        true_shift.append(torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, s].float()) for i in range(0, R, 16)]))
    true_shift = torch.stack(true_shift, 1)                                        # [R,5,16]
    true_off = offsets(true_shift)                                                 # [R,5,16,2]
    agree = (true_off == true_off[:, :1]).all(-1).all(1)                          # [R,16]
    states = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0, :15]], 1).float()
    cats = torch.stack([move_table(states[:, k])[0] for k in range(16)], 1)      # [R,16,17]: 0 moved, 1 blocked
    act_cls = torch.where((fa >= 1) & (fa <= 4), cats.gather(2, fa[..., None].clamp(max=16))[..., 0], torch.full_like(fa, 2))
    out_dir = HERE / "evals"
    for path in [Path(p) for p in sys.argv[1:]]:
        world, st = T.load_world(path, device)
        name = st["name"]
        batch = 16 if world.backbone_kind == "full" else 4
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        for i in range(0, R, batch):
            b = min(batch, R - i)
            c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
            frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
            for k in range(H):
                w = 4 if k == 0 else 5
                g = T.step(world, torch.stack(frames[-w:], 1), torch.stack(hist[-(w - 1):] + [fk[:, k]], 1), device, config)
                gen[i:i + b, k] = g.half(); frames.append(g); hist.append(fk[:, k])
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_shift = torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)])  # [R,16]
        img_off = offsets(img_shift)
        aligned = (img_off == true_off[:, 0]).all(-1)                              # [R,16]
        valid = alive & agree
        # excess per token, then split
        sums = {g: {"map": torch.zeros(H), "hud": torch.zeros(H), "n": torch.zeros(H)} for g in ("aligned", "wrong")}
        realigned_map, wrong_map_cov = torch.zeros(H), torch.zeros(H)
        for i in range(0, R, 16):
            b = min(16, R - i)
            x = fut5[i:i + b].float(); mu = x.mean(1)
            s2 = ((x - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)
            g = gen[i:i + b].float()
            ex = ((g - mu) ** 2).sum(-1) - s2 / S                                  # [b,16,81]
            for gname, m in (("aligned", aligned[i:i + b] & valid[i:i + b]), ("wrong", ~aligned[i:i + b] & valid[i:i + b])):
                sums[gname]["map"] += (ex[..., :63] * m[..., None]).sum((0, 2))
                sums[gname]["hud"] += (ex[..., 63:] * m[..., None]).sum((0, 2))
                sums[gname]["n"] += m.sum(0).float()
            for j in range(b):
                for k in range(H):
                    if not (valid[i + j, k] and not aligned[i + j, k]):
                        continue
                    d = (true_off[i + j, 0, k] - img_off[i + j, k]).tolist()
                    # imagined view centre is at img offset; the true one at true offset: true cell (r,c) = imagined (r+dr, c+dc)
                    shifted, ok = shift_map(g[j, k], int(d[0]), int(d[1]))
                    e_map = ((shifted - mu[j, k, :63]) ** 2).sum(-1) - s2[j, k, :63] / S
                    realigned_map[k] += float(e_map[ok].sum()) * 63 / max(int(ok.sum()), 1)
                    wrong_map_cov[k] += float(ex[j, k, :63].sum())
        n_valid = valid.sum(0).float().clamp(min=1)
        first = torch.where((~aligned & valid).any(1), (~aligned & valid).float().argmax(1), torch.full((R,), -1))
        cause = {}
        for r in torch.where(first >= 0)[0].tolist():
            k = int(first[r])
            key = f"{['moved', 'blocked', 'other'][int(act_cls[r, k])]}:true{int(true_shift[r, 0, k])}_img{int(img_shift[r, k])}"
            cause[key] = cause.get(key, 0) + 1
        res = {"world": name, "V": V, "valid_by_depth": n_valid.tolist(),
               "wrong_rate": (sums["wrong"]["n"] / n_valid).tolist(),
               "first_wrong_depth_hist": torch.bincount(first[first >= 0], minlength=H).tolist(),
               "roots_ever_wrong": int((first >= 0).sum()),
               "first_wrong_cause": dict(sorted(cause.items(), key=lambda kv: -kv[1])[:20]),
               "excess_per_valid_root": {gname: {"map": (sums[gname]["map"] / n_valid / V).tolist(),
                                                 "hud": (sums[gname]["hud"] / n_valid / V).tolist()} for gname in sums},
               "wrong_map_excess_before_after_realign": {"before": (wrong_map_cov / n_valid / V).tolist(),
                                                         "after": (realigned_map / n_valid / V).tolist()}}
        k16 = H - 1
        map_total = float(sums["aligned"]["map"][k16] + sums["wrong"]["map"][k16])
        wrong_share = float(sums["wrong"]["map"][k16]) / map_total if map_total else 0.0
        removed = 1 - float(realigned_map[k16]) / float(wrong_map_cov[k16]) if float(wrong_map_cov[k16]) else 0.0
        res["readings"] = {"map_excess_share_wrong_16": wrong_share, "realign_removes_16": removed,
                           "position_dominated": wrong_share >= 0.5 and removed >= 0.5,
                           "content_dominated": (1 - wrong_share) >= 0.5}
        (out_dir / f"driftanat_{name}.json").write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps({"world": name, **res["readings"], "wrong_rate_16": res["wrong_rate"][k16],
                          "roots_ever_wrong": res["roots_ever_wrong"]}), flush=True)
        del world
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE.parent / "20260926_diagnosis"))
    sys.path.insert(0, str(HERE.parent / "20260921_readout_ladder"))
    main()
