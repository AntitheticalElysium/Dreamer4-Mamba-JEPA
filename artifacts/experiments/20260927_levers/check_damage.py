"""Decision check (2026-10-02): why does imagination never draw the damage? (check_transfer_subst: swapping ONLY the 18 HUD tokens of
the imagined step-1 state for the real successor's closes 99.8-99.9% of the real-head transfer gap at all four worlds, the 3 x 3
around the player 5-6%; check_hud_danger: imagined health ignores the deadly action.)
Diagnosis futures, sample 0, simulator truth (health = visible[1512] * 9). Transitions t -> t+1 (alive at t+1) where the true health
DROPS ("damage"), vs where it is unchanged. Per world:
  caught     the world's predicted next frame shows a drop: teval HUD probe health(pred) < health(current true frame) - 0.5 / 9
             -- teacher-forced (true window) and self-fed (the world's rollout, aligned steps)
  false      the same on unchanged transitions (a drop drawn where none happened)
Data-only: damage rate per transition; HUD squared token change t -> t+1 on damage vs unchanged transitions (the token size of a
lost heart vs ordinary frame-to-frame HUD noise).
Reading, declared before running: damage_copied = teacher-forced caught <= 0.2 in every world (the drop is not drawn even from true
inputs); damage_small = median HUD token change on damage <= 2 x the unchanged median.
Usage: check_damage.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_decision_step as CD
DA, SD, T = CD.DA, CD.SD, CD.T
H = CD.H


def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    root = ctx[:, -1]
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()
    hp = allv[:, :, 1512] * 9                                                          # [R,17]
    frames_true = torch.cat([root[:, None], fut5[:, 0]], 1)                            # [R,17,81,192]
    prev = torch.cat([root[:, None], fut5[:, 0, :-1]], 1)
    ts0 = torch.cat([estimate(prev[i:i + 16].float(), fut5[i:i + 16, 0].float()) for i in range(0, R, 16)])
    true_off = DA.offsets(ts0).long()
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().flatten(-2))[..., 0] * 9
    drop = (hp[:, 1:] < hp[:, :-1] - 0.5) & alive                                      # [R,16] transition k: frame k -> k+1
    same = (hp[:, 1:] == hp[:, :-1]) & alive
    dtok = ((frames_true[:, 1:, 63:81].float() - frames_true[:, :-1, 63:81].float()) ** 2).sum((-1, -2))
    out = {"damage_transitions": int(drop.sum()), "unchanged_transitions": int(same.sum()),
           "damage_rate": float(drop.sum() / alive.sum()),
           "hud_token_change_median": {"damage": float(dtok[drop].median()), "unchanged": float(dtok[same].median())}}
    h_true = torch.stack([health(frames_true[:, k]) for k in range(17)], 1)            # probe on true frames [R,17]
    out["probe_check"] = {"true_drop_seen_by_probe": float((h_true[:, 1:] < h_true[:, :-1] - 0.5)[drop].float().mean()),
                          "false_drop_by_probe": float((h_true[:, 1:] < h_true[:, :-1] - 0.5)[same].float().mean())}
    print(json.dumps(out), flush=True)
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        name = st["name"]
        tf = torch.empty(R, H)
        sf = torch.empty(R, H)
        gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
        with torch.no_grad():
            for i in range(0, R, 64):
                b = min(64, R - i)
                c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
                frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
                tfr = [c4[:, j] for j in range(4)]
                for k in range(H):
                    w = 4 if k == 0 else 5
                    acts = torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                    g = T.step(world, torch.stack(frames[-w:], 1), acts, device, config)
                    t = T.step(world, torch.stack(tfr[-w:], 1), acts, device, config)
                    gen[i:i + b, k] = g.half(); tf[i:i + b, k] = health(t)
                    frames.append(g); hist.append(fk[:, k]); tfr.append(fut5[i:i + b, 0, k].float())
        sf = health(gen.float())
        gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
        img_off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].float(), gen[j:j + 32].float()) for j in range(0, R, 32)])).long()
        aligned = torch.cat([torch.ones(R, 1, dtype=torch.bool), (img_off == true_off).all(-1)[:, :-1]], 1)   # aligned before step k
        h_cur = h_true[:, :-1]                                                         # true current health (probe)
        h_cur_self = torch.cat([h_true[:, :1], sf[:, :-1]], 1)                         # imagined current health
        r = {}
        for mode, pred, cur, extra in (("teacher", tf, h_cur, torch.ones_like(drop)), ("selffed", sf, h_cur_self, aligned)):
            pdrop = pred < cur - 0.5
            r[mode] = {"caught": float(pdrop[drop & extra].float().mean()), "n_damage": int((drop & extra).sum()),
                       "false_drop": float(pdrop[same & extra].float().mean())}
        r["readings_part"] = {"teacher_caught_le_0.2": r["teacher"]["caught"] <= 0.2}
        out[name] = r
        print(json.dumps({name: r}), flush=True)
        del world; torch.cuda.empty_cache()
    ws = [k for k in out if isinstance(out[k], dict) and "teacher" in out[k]]
    out["readings"] = {"damage_copied": all(out[k]["teacher"]["caught"] <= 0.2 for k in ws),
                       "damage_small": out["hud_token_change_median"]["damage"] <= 2 * out["hud_token_change_median"]["unchanged"]}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
