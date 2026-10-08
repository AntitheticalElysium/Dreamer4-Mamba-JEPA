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
Added 2026-10-03 before the first run, reported only (check_damage_rule: a zombie hit lands with P 0.97 after a fresh arrival,
0.48 when a zombie was already beside the player and no hit is in the 4-frame window, 0.03 after a hit in the window): teacher-forced
caught / drawn-without-a-hit by that visible history (sample 0, k >= 3, drops of >= 2).
Risk-suite additions (2026-10-03, reported): `--window W` (rollout context; default 5 = the convention above, frames available
min(4 + k, W)); health-change accuracy by true change class (<= -2, -1, 0, >= +1; predicted class from pred - current with
+-0.5 / -1.5 cuts), teacher-forced; each visible-history case's Bayes hit rate (check_damage_rule, all 5 samples) beside the catch.
Reading, declared before running: damage_copied = teacher-forced caught <= 0.2 in every world (the drop is not drawn even from true
inputs); damage_small = median HUD token change on damage <= 2 x the unchanged median.
Usage: check_damage.py <world.pt> ...
Resume (2026-10-04): same command reuses hash-bound completed root batches (teacher health and self-fed tokens) and
completed world results in artifacts/eda/frozen_eval_resume_v1. Input/label tensors, numeric sources, weights and window
are bound; mismatches use a different contract, corrupt committed states fail. Model mathematics/metrics are unchanged.
Result (2026-10-03; sample 0: 337 damage transitions, rate 0.021; s7 18k / s7 36k / s8 18k / s8 36k):
  damage_copied TRUE: teacher-forced caught 0.018 / 0.030 / 0.021 / 0.050 (false drops 0.009-0.014); self-fed 0.004-0.009.
  damage_small FALSE: HUD squared token change on damage median 74.9 vs 2.1 unchanged (36x); the probe sees every true drop.
  By visible history (teacher, k >= 3): fresh arrivals (hit P 0.97 from the window) caught 0 / 0 / 0 / 1 of 31; beside without a
  hit in the window caught 0.023 / 0.041 / 0.027 / 0.073 of 220 hits but drawn on 0.080 / 0.112 / 0.085 / 0.134 of 224 no-hit
  transitions (drawn drops do not track hits); after a hit in the window drawn on 0.17-0.24 of no-hit transitions.
  So the hit is copied even where the input predicts it (the rarity pathology of the DO consequences, slowly shrinking with
  budget), and where it is a coin flip the L1 median removes it anyway.
50k extensions (lane49; 42k / 48k / 50k snapshots): teacher caught s7 0.101 / 0.033 / 0.042, s8 0.074 / 0.042 / 0.045; fresh
  arrivals 0 (s7) / 1 (s8) of 31 at every snapshot; beside, drawn on hits 0.05-0.19 vs on no-hit transitions 0.10-0.27. Unlike the
  placements (consfit: stage-like jumps 42k -> 50k), no hit mode is learned by 50k.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import h16_resume as RSM
import check_decision_step as CD
import check_damage_rule as DR
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
    MA = DR.masks(meta)
    M = {k: v[:, 0] for k, v in MA.items()}                                            # sample 0, transition k: frame k -> k+1
    base = M["valid"] & M["k3"]
    cases = {"fresh": M["adjacent"] & ~M["win"] & ~M["adjwin"], "beside_no_hit": M["adjacent"] & ~M["win"] & M["adjwin"],
            "hit_in_window": M["adjacent"] & M["win"]}
    use = MA["valid"] & MA["k3"]
    bayes = {"fresh": MA["adjacent"] & ~MA["win"] & ~MA["adjwin"], "beside_no_hit": MA["adjacent"] & ~MA["win"] & MA["adjwin"],
             "hit_in_window": MA["adjacent"] & MA["win"]}
    out["bayes_hit_rate"] = {c: round(float(MA["drop2"][use & q].float().mean()), 4) for c, q in bayes.items()}
    args = sys.argv[1:]
    window = int(args[args.index("--window") + 1]) if "--window" in args else 5
    paths = [a for i, a in enumerate(args) if a != "--window" and (i == 0 or args[i - 1] != "--window")]
    dclass = lambda d: torch.bucketize(d, torch.tensor([-1.5, -0.5, 0.5]))            # 0: <= -2, 1: -1, 2: 0, 3: >= +1
    for path in paths:
        name = torch.load(path, map_location="cpu", weights_only=False)["name"]
        store = RSM.frozen_eval_store(path, name + f"__damage_w{window}", {"window": window, "task": "damage"},
                                     {**{f"cache_{k}": v for k, v in cache.items() if isinstance(v, torch.Tensor)},
                                      **{f"meta_{k}": v for k, v in meta.items() if isinstance(v, torch.Tensor)},
                                      "future5": fut5, "visible": allv, "alive": alive,
                                      "train_roots": train_roots, "train_seeds": train_seeds,
                                      **{f"mask_{k}": v for k, v in MA.items()}}, [Sp.CHECKPOINT])
        with store.lock():
            completed = store.load("result")
            if completed is not None:
                out[name] = completed
                print(json.dumps({name: completed}), flush=True)
                continue
            world, st = T.load_world(Path(path), device)
            tf = torch.empty(R, H)
            sf = torch.empty(R, H)
            gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
            bs = 16 if getattr(world, "backbone_kind", "full") in ("fmamba", "fcanvas") else 64   # per-token SSM memory (2026-10-04)
            with torch.no_grad():
                for i in range(0, R, bs):
                    b = min(bs, R - i)
                    saved = store.load(f"batch_{i}")
                    if saved is not None:
                        tf[i:i + b], gen[i:i + b] = saved["teacher"], saved["generated"]
                        continue
                    c4, a3, fk = ctx[i:i + b].float(), ca[i:i + b], fa[i:i + b]
                    frames, hist = [c4[:, j] for j in range(4)], [a3[:, j] for j in range(3)]
                    tfr = [c4[:, j] for j in range(4)]
                    for k in range(H):
                        w = min(4 + k, window)
                        acts = torch.stack(hist[-(w - 1):] + [fk[:, k]], 1)
                        g = T.step(world, torch.stack(frames[-w:], 1), acts, device, config)
                        t = T.step(world, torch.stack(tfr[-w:], 1), acts, device, config)
                        gen[i:i + b, k] = g.half(); tf[i:i + b, k] = health(t)
                        frames.append(g); hist.append(fk[:, k]); tfr.append(fut5[i:i + b, 0, k].float())
                    store.save(f"batch_{i}", {"teacher": tf[i:i + b].clone(), "generated": gen[i:i + b].clone()}, 1)
            sf = torch.stack([health(gen[:, k].float()) for k in range(H)], 1)
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
            pd_t = tf < h_cur - 0.5
            r["teacher_by_history"] = {c: {"n_hit": int((base & q & M["drop2"]).sum()),
                                           "caught": float(pd_t[base & q & M["drop2"]].float().mean()),
                                           "n_no_hit": int((base & q & ~M["drop2"]).sum()),
                                           "drawn_without_hit": float(pd_t[base & q & ~M["drop2"]].float().mean())} for c, q in cases.items()}
            true_c, pred_c = dclass(hp[:, 1:] - hp[:, :-1]), dclass(tf - h_cur)
            r["health_change_accuracy"] = {name_: {"n": int((alive & (true_c == c)).sum()),
                                                  "acc": round(float((pred_c == c)[alive & (true_c == c)].float().mean()), 4)}
                                           for c, name_ in enumerate(("le-2", "-1", "0", "ge+1"))}
            r["window"] = window
            r["readings_part"] = {"teacher_caught_le_0.2": r["teacher"]["caught"] <= 0.2}
            store.save("result", r, 1)
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
