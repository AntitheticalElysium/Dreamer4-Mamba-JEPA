"""Mamba diagnosis (2026-10-04, the user's request: find the exact reason Mamba shows no difference). How much of its input
history does each world USE? Every per-tile Mamba comparison so far (stage 2 6k: per-token Mamba 0.245 vs factored attention
0.243 one-step; E17 stage 1 36k: -0.002 one-step, depth 16 ns) trained on 6-frame windows and was evaluated by re-scanning the
last <= 5 frames each step (teval convention). The 2026-09-30 position profile found the 18k Mamba world WORSE with more true
context (w1 0.0398, w4 0.0483).
Diagnosis futures, sample 0, TRUE frames (teacher-forced), transitions k >= 3 (so 5 true frames precede every target, check_
damage_rule's history cases defined): the world predicts frame k+1 from the last w true frames, w = 1..5. Reported per w:
squared error by token class (map 0-62, player 31, HUD 63-80) and the hit catch / false-drop rate (check_damage's probe and cuts)
with the hit's visible-history case.
Readings, declared before running (worlds: attention and Mamba 36k teacher, seed 7):
  uses_history      error(w = 5) at least 5% below error(w = 1) on the map or the HUD, per world
  mamba_uses_more   Mamba's relative gain from w = 1 to 5 exceeds attention's by >= 5 points on the HUD (the cooldown lives there)
Results (2026-10-04, 13,022 transitions; log check_context.log), error per transition w1 -> w5:
  attention  map 213.7 -> 206.1 (-3.6%)  HUD 25.72 -> 29.64 (+15.2%)  player 0.635 -> 0.593; hits drawn only at w5: catch 3.2%,
             false drops 1.1%, fresh 0%
  Mamba      map 218.9 -> 209.2 (-4.5%; w4 205.8, -6.0%)  HUD 25.72 -> 29.50 (+14.7%); w5 only: catch 6.1%, false 0.8%, fresh 0%
  uses_history FALSE for both, mamba_uses_more FALSE. Neither world uses its <= 5-frame history beyond ~4-6% on the map.
  Drops appear only at w5, where window length and "last frame on time row 4" coincide: separated by check_position.py.
Usage: check_context.py <world.pt> ...
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_damage_rule as DR  # noqa: E402
import check_decision_step as CD  # noqa: E402
SD, T = CD.SD, CD.T
H = CD.H


def main():
    from d4mj.config import config_from_dict
    import spatial as Sp
    device = torch.device("cuda")
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = len(meta["seed"])
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9
    frames = torch.cat([cache["ctx"].cpu(), fut5[:, 0].cpu()], 1)                    # [R, 20, 81, 192]: 3 = root
    acts = torch.cat([cache["ctx_a"].cpu(), cache["fut_a"].cpu()], 1)                  # [R, 19]
    M0 = {k: v[:, 0] for k, v in DR.masks(meta).items()}
    use = M0["valid"] & M0["k3"]
    cases = {"fresh": M0["adjacent"] & ~M0["win"] & ~M0["adjwin"], "beside_no_hit": M0["adjacent"] & ~M0["win"] & M0["adjwin"]}
    out = {}
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        res = {}
        for w in range(1, 6):
            err = {"map": 0.0, "player": 0.0, "hud": 0.0}; n = 0
            drawn = torch.zeros(R, H, dtype=torch.bool)
            for k in range(3, H):
                c = 3 + k
                for i in range(0, R, 64):
                    win, wa = frames[i:i + 64, c + 1 - w:c + 1].float(), acts[i:i + 64, c + 1 - w:c + 1]
                    pred = T.step(world, win, wa, device, config)
                    tgt = frames[i:i + 64, c + 1].float()
                    m = use[i:i + 64, k]
                    e = ((pred - tgt) ** 2).sum(-1)[m]
                    err["map"] += float(e[:, :63].sum()); err["player"] += float(e[:, 31].sum()); err["hud"] += float(e[:, 63:].sum())
                    n += int(m.sum())
                    drawn[i:i + 64, k] = health(pred) < health(frames[i:i + 64, c].float()) - 1.5
            hit = M0["drop2"]
            res[f"w{w}"] = {**{f"err_{k}": v / max(n, 1) for k, v in err.items()}, "transitions": n,
                            "hit_caught": float(drawn[use & hit].float().mean()), "false_drop": float(drawn[use & ~hit].float().mean()),
                            **{f"caught_{c}": float(drawn[use & hit & q].float().mean()) for c, q in cases.items()}}
            print(json.dumps({st["name"]: {f"w{w}": res[f"w{w}"]}}), flush=True)
        for cls in ("map", "hud"):
            res[f"gain_w1_to_w5_{cls}"] = 1 - res["w5"][f"err_{cls}"] / res["w1"][f"err_{cls}"]
        out[st["name"]] = res
        del world; torch.cuda.empty_cache()
    rd = {k: (v["gain_w1_to_w5_map"] >= 0.05 or v["gain_w1_to_w5_hud"] >= 0.05) for k, v in out.items()}
    names = list(out)
    att = [k for k in names if "fmamba" not in k]; mam = [k for k in names if "fmamba" in k]
    if att and mam:
        rd["mamba_uses_more"] = out[mam[0]]["gain_w1_to_w5_hud"] - out[att[0]]["gain_w1_to_w5_hud"] >= 0.05
    out["readings"] = {"uses_history": rd}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
