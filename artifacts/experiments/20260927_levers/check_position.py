"""Position or context? (2026-10-04, from check_context). check_context found both 36k worlds draw health drops ONLY with a
5-frame window (hit catch attention 3.2% / Mamba 6.1%, false drops 1.1% / 0.8%; 0 for 1-4 frames) and the HUD error rising 15%
there. The time table is start-aligned (TWorld.inputs adds time[:t]), so "5 frames of context" and "last frame on time row 4"
are the same thing in every run so far. Measured in the pool (raw, 6-frame windows): 78% of the training health drops <= -2 sit
at row 4 (main 339 + terminal 6,172 vs ~460 at each other row): the end-aligned terminal windows put 90% of their deaths on the
last transition.
Substitution: the same windows placed on different time rows (the world's time parameter temporarily re-indexed; weights
unchanged). Diagnosis futures, sample 0, TRUE frames, transitions k >= 3; probe, cuts and cases as check_context.
  w4_at0   last 4 frames on rows 0-3 (check_context's w4)       w4_at1   the same 4 frames on rows 1-4
  w5_at0   last 5 frames on rows 0-4 (the evaluation convention) w1_at0 / w1_at4   the last frame alone on row 0 / row 4
Readings, declared before running (per world: attention and Mamba 36k teacher, seed 7):
  position_shortcut  drawn-drop rate (all used transitions) of w4_at1 >= half of w5_at0's AND w4_at0's <= a fifth of w5_at0's:
                     the drops follow the time row, not the fifth frame of history
  row4_alone         w1_at4's drawn-drop rate >= half of w5_at0's (one frame on row 4 suffices)
Usage: check_position.py <world.pt> ...
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
CONDITIONS = {"w4_at0": (4, 0), "w4_at1": (4, 1), "w5_at0": (5, 0), "w1_at0": (1, 0), "w1_at4": (1, 4)}   # (frames, first row)


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
    use, hit = M0["valid"] & M0["k3"], M0["drop2"]
    cases = {"fresh": M0["adjacent"] & ~M0["win"] & ~M0["adjwin"], "beside_no_hit": M0["adjacent"] & ~M0["win"] & M0["adjwin"]}
    out = {}
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        saved = world.time.data.clone()
        res = {}
        for name, (w, row) in CONDITIONS.items():
            world.time.data = saved.clone()
            world.time.data[:w] = saved[row:row + w]                                    # window frame j sits on row (row + j)
            drawn = torch.zeros(R, H, dtype=torch.bool)
            for k in range(3, H):
                c = 3 + k
                for i in range(0, R, 64):
                    pred = T.step(world, frames[i:i + 64, c + 1 - w:c + 1].float(), acts[i:i + 64, c + 1 - w:c + 1], device, config)
                    drawn[i:i + 64, k] = health(pred) < health(frames[i:i + 64, c].float()) - 1.5
            res[name] = {"drawn_rate": float(drawn[use].float().mean()), "hit_caught": float(drawn[use & hit].float().mean()),
                         "false_drop": float(drawn[use & ~hit].float().mean()), "hits": int((use & hit).sum()),
                         **{f"caught_{c}": float(drawn[use & hit & q].float().mean()) for c, q in cases.items()}}
            print(json.dumps({st["name"]: {name: res[name]}}), flush=True)
        world.time.data = saved
        r = {k: v["drawn_rate"] for k, v in res.items()}
        res["readings"] = {"position_shortcut": r["w4_at1"] >= r["w5_at0"] / 2 and r["w4_at0"] <= r["w5_at0"] / 5,
                           "row4_alone": r["w1_at4"] >= r["w5_at0"] / 2}
        out[st["name"]] = res
        del world; torch.cuda.empty_cache()
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
