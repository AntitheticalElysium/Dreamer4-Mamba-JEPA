"""E16 evaluation (readings predeclared in NOTEBOOK.md 2026-10-03, commit 7e353eba). Diagnosis futures, sample 0, transitions
k >= 3 (check_damage_rule's visible-history cases need the 4-frame window inside the trajectory), simulator health truth.
  posterior   teacher-forced window, the stage-A posterior's Delta from the TRUE next frame: drawn health-change accuracy by
              class (<= -2, -1, 0, >= +1; check_damage's cuts) -> e16_health_drawable (>= 0.8 for -2, -1 and +1)
  prior       teacher-forced window, Delta SAMPLED by the stage-B prior given the true history (posterior codes for the past
              transitions), M samples per step: drawn hit frequency per visible-history case vs check_damage_rule's Bayes rate
              -> e16_hits_calibrated (within +-0.15 for fresh / already beside / recent hit), and with no zombie beside the
              post-move player -> e16_no_false_hits (<= 0.01)
  selffed     the world's own sampled rollout (one sample), drawn hit frequency per case on steps whose imagined view is still
              aligned with the true one (check_damage's alignment rule); reported
  delta_use   (added 2026-10-03 before the first run, reported) teacher-forced squared error of the decoded next frame with the
              posterior's Delta vs with Delta zeroed, per token class: HUD (63-80), player (31), cells entering the view on scroll
              steps (scroll.estimate on the true pair), the rest of the map: what the channel carries
A drawn hit = the HUD probe's health on the predicted frame below the current health (true current for teacher-forced, imagined
for self-fed) by more than 1.5 (check_damage's class cut for <= -2).
First s7 run (2026-10-04 12:10, log e16_check_s7_v1_lastframe.log) set Delta on the last window frame only; training conditions
every frame (dworld.Posterior.forward), fixed here. A CPU rerun of the fixed version on 32 roots gives the same posterior -2
accuracy (0/10), and the decoder trace on pool windows (NOTEBOOK 2026-10-04) shows why it is not an evaluation artifact: the
decoder draws health changes only on the death transitions of end-aligned terminal windows; on ordinary hits it copies.
Usage: check_e16.py <stage-B prior .pt> [--samples M]
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_damage_rule as DR  # noqa: E402
import check_decision_step as CD  # noqa: E402
import dworld as DW  # noqa: E402
DA, SD, T = CD.DA, CD.SD, CD.T
H = CD.H


def step(world, post, frames, acts, codes, device, config):
    """frames [B,w,81,192], acts [B,w], codes [B,w,4]: the code of the transition LEAVING each window frame (training conditions
    every frame on its own; FIXED 2026-10-04: the first version set only the last frame's) -> next frame [B,81,192]"""
    b, w = frames.shape[:2]
    with torch.no_grad():
        delta = post.condition(post.quantizer.embed(codes.to(device).flatten(0, 1))).view(b, w, 81, DW.S.D)
    world.delta = delta
    out = T.step(world, frames, acts, device, config)
    world.delta = None
    return out


def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    import os
    device = torch.device(os.environ.get("E16_DEVICE", "cuda"))                     # E16_DEVICE / E16_ROOTS: tests only
    args = sys.argv[1:]
    M = int(args[args.index("--samples") + 1]) if "--samples" in args else 8
    prior_path = Path(args[0])
    st_b = torch.load(prior_path, map_location="cpu", weights_only=False)
    world, post, _ = DW.load_a(Path(st_b["stage_a"]), device)
    prior = DW.Prior().to(device); prior.load_state_dict(st_b["prior"]); prior.eval()
    config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    meta, train_roots, train_seeds = T.split()
    fut5, _ = SD.token_cache(device)
    cache = T.build_cache("raw", device)
    R = int(os.environ.get("E16_ROOTS", len(meta["seed"])))
    ctx, ca, fa = cache["ctx"][:R], cache["ctx_a"][:R], cache["fut_a"][:R]
    fut5 = fut5[:R]
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9          # probes are CPU ridges
    MA = {k: v[:R] for k, v in DR.masks(meta).items()}
    M0 = {k: v[:, 0] for k, v in MA.items()}
    use = MA["valid"] & MA["k3"]
    cases = {"fresh": lambda m: m["adjacent"] & ~m["win"] & ~m["adjwin"], "beside_no_hit": lambda m: m["adjacent"] & ~m["win"] & m["adjwin"],
             "hit_in_window": lambda m: m["adjacent"] & m["win"], "no_zombie_beside": lambda m: ~m["adjacent"]}
    bayes = {c: float(MA["drop2"][use & f(MA)].float().mean()) for c, f in cases.items()}
    allv = torch.cat([meta["root_visible"][:R, None], meta["future_visible"][:R, 0]], 1).float()
    hp = allv[:, :, 1512] * 9
    alive = ~meta["future_dead"][:R].cumsum(2).bool().any(1)
    true_frames = torch.cat([ctx.cpu(), fut5[:, 0].cpu()], 1)                        # [R, 4 + 16, 81, 192]
    acts_all = torch.cat([ca.cpu(), fa.cpu()], 1)                                                # [R, 3 + 16]: action t leads frame t -> t+1
    post_h = torch.empty(R, H); prior_hits = torch.zeros(R, H); self_h = torch.empty(R, H)
    from scroll import SHIFTS
    use_err = {c: [0.0, 0.0] for c in ("hud", "player", "entering", "map_rest")}             # [with posterior Delta, Delta zeroed]
    gen = torch.empty(R, H, 81, 192, dtype=torch.float16)
    g = torch.Generator(device=device).manual_seed(20261003)
    with torch.no_grad():
        for i in range(0, R, 32):
            b = min(32, R - i)
            tf = true_frames[i:i + b].float().to(device); ac = acts_all[i:i + b].to(device)
            # posterior codes for every true transition (context 3 + future 16)
            pc = post.quantizer(post.encode(tf[:, :-1].flatten(0, 1), ac.flatten(), tf[:, 1:].flatten(0, 1)))[1].view(b, -1, DW.K)
            imagined = [tf[:, j] for j in range(4)]; icodes = [pc[:, j] for j in range(3)]
            for k in range(H):
                c = 3 + k                                                            # index of the current frame in tf
                w = min(c + 1, 5)
                win, wa = tf[:, c + 1 - w:c + 1], ac[:, c + 1 - w:c + 1]
                with_d = step(world, post, win, wa, pc[:, c + 1 - w:c + 1], device, config)
                post_h[i:i + b, k] = health(with_d).cpu()
                world.delta = None
                no_d = T.step(world, win, wa, device, config)
                nxt_true = tf[:, c + 1].cpu()
                sh = estimate(tf[:, c].cpu(), nxt_true)
                ent = torch.zeros(b, 81, dtype=torch.bool)
                for j in range(b):
                    dr, dc = SHIFTS[int(sh[j])]
                    if dr: ent[j, [(6 if dr == 1 else 0) * 9 + cc for cc in range(9)]] = True
                    if dc: ent[j, [rr * 9 + (8 if dc == 1 else 0) for rr in range(7)]] = True
                live = alive[i:i + b, k] if k < H else torch.ones(b, dtype=torch.bool)
                masks = {"hud": torch.zeros(b, 81, dtype=torch.bool), "player": torch.zeros(b, 81, dtype=torch.bool), "entering": ent}
                masks["hud"][:, 63:81] = True; masks["player"][:, 31] = True
                masks["map_rest"] = ~(masks["hud"] | masks["player"] | ent)
                e1, e0 = ((with_d - nxt_true) ** 2).sum(-1), ((no_d - nxt_true) ** 2).sum(-1)       # [b,81]
                for cname, m in masks.items():
                    mm = m & live[:, None]
                    use_err[cname][0] += float(e1[mm].sum()); use_err[cname][1] += float(e0[mm].sum())
                hist_s, hist_a, hist_c = tf[:, max(0, c + 1 - DW.BLOCKS):c + 1], ac[:, max(0, c + 1 - DW.BLOCKS):c + 1], pc[:, max(0, c + 1 - DW.BLOCKS):c]
                cur = health(tf[:, c]).cpu()
                for _ in range(M):
                    codes, _ = prior.sample(hist_s, hist_a, hist_c, generator=g)
                    cw = torch.cat([pc[:, c + 1 - w:c], codes[:, None]], 1)
                    prior_hits[i:i + b, k] += (health(step(world, post, win, wa, cw, device, config).to(device)).cpu() < cur - 1.5).float() / M
                iw = min(len(imagined), 5)
                isn = torch.stack(imagined[-DW.BLOCKS:], 1); ian = ac[:, max(0, c + 1 - len(imagined[-DW.BLOCKS:])):c + 1]
                scodes, _ = prior.sample(isn, ian, torch.stack(icodes[-(isn.shape[1] - 1):], 1), generator=g)
                cw = torch.stack(icodes[len(icodes) - (iw - 1):] + [scodes], 1)
                nxt = step(world, post, torch.stack(imagined[-iw:], 1), ac[:, c + 1 - iw:c + 1], cw, device, config).to(device)
                gen[i:i + b, k] = nxt.half().cpu(); self_h[i:i + b, k] = health(nxt).cpu()
                imagined.append(nxt); icodes.append(scodes)
    dclass = lambda d: torch.bucketize(d, torch.tensor([-1.5, -0.5, 0.5]))
    h_true = torch.stack([health(true_frames[:, 3 + k].float()).cpu() for k in range(H + 1)], 1)          # probe on true frames
    true_c, post_c = dclass(hp[:, 1:] - hp[:, :-1]), dclass(post_h - h_true[:, :-1])
    out = {"samples": M, "bayes": bayes,
           "delta_use": {c: {"err_with_delta": v[0], "err_delta_zeroed": v[1], "reduction": (v[1] - v[0]) / max(v[1], 1e-9)}
                         for c, v in use_err.items()}}
    out["posterior_health_accuracy"] = {n: {"n": int((alive & (true_c == ci)).sum()), "acc": round(float((post_c == ci)[alive & (true_c == ci)].float().mean()), 4)}
                                        for ci, n in enumerate(("le-2", "-1", "0", "ge+1"))}
    base = M0["valid"] & M0["k3"]
    out["prior_hit_rate"] = {c: round(float(prior_hits[base & f(M0)].mean()), 4) for c, f in cases.items()}
    gprev = torch.cat([ctx[:, -1:].float().cpu(), gen[:, :-1].float()], 1)
    img_off = DA.offsets(torch.cat([estimate(gprev[j:j + 32], gen[j:j + 32].float()) for j in range(0, R, 32)])).long()
    prev = torch.cat([ctx[:, -1:].cpu(), fut5[:, 0, :-1].cpu()], 1)
    true_off = DA.offsets(torch.cat([estimate(prev[j:j + 16].float(), fut5[j:j + 16, 0].float().cpu()) for j in range(0, R, 16)])).long()
    aligned = torch.cat([torch.ones(R, 1, dtype=torch.bool), (img_off == true_off).all(-1)[:, :-1]], 1)
    self_cur = torch.cat([h_true[:, :1], self_h[:, :-1]], 1)
    self_hit = (self_h < self_cur - 1.5).float()
    out["selffed_hit_rate"] = {c: round(float(self_hit[base & aligned & f(M0)].mean()), 4) for c, f in cases.items()}
    out["readings"] = {"e16_health_drawable": all(out["posterior_health_accuracy"][n]["acc"] >= 0.8 for n in ("le-2", "-1", "ge+1")),
                       "e16_hits_calibrated": all(abs(out["prior_hit_rate"][c] - bayes[c]) <= 0.15 for c in ("fresh", "beside_no_hit", "hit_in_window")),
                       "e16_no_false_hits": out["prior_hit_rate"]["no_zombie_beside"] <= 0.01}
    print(json.dumps(out))


if __name__ == "__main__":
    sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
    sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
    main()
