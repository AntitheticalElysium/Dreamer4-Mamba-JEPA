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
A drawn hit = the HUD probe's health on the predicted frame below the current health (true current for teacher-forced, imagined
for self-fed) by more than 1.5 (check_damage's class cut for <= -2).
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
    """frames [B,w,81,192], acts [B,w], codes [B,4] for the last transition -> next frame [B,81,192]"""
    b, w = frames.shape[:2]
    delta = torch.zeros(b, w, 81, DW.S.D, device=device)
    with torch.no_grad():
        delta[:, -1] = post.condition(post.quantizer.embed(codes.to(device)))
    world.delta = delta
    out = T.step(world, frames, acts, device, config)
    world.delta = None
    return out


def main():
    from d4mj.config import config_from_dict
    from scroll import estimate
    import spatial as Sp
    device = torch.device("cuda")
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
    R = len(meta["seed"])
    ctx, ca, fa = cache["ctx"], cache["ctx_a"], cache["fut_a"]
    probes = T.Probes(cache, meta, train_roots, train_seeds)
    health = lambda tok: probes.hud(tok[..., 63:81, :].float().cpu().flatten(-2))[..., 0] * 9          # probes are CPU ridges
    MA = DR.masks(meta)
    M0 = {k: v[:, 0] for k, v in MA.items()}
    use = MA["valid"] & MA["k3"]
    cases = {"fresh": lambda m: m["adjacent"] & ~m["win"] & ~m["adjwin"], "beside_no_hit": lambda m: m["adjacent"] & ~m["win"] & m["adjwin"],
             "hit_in_window": lambda m: m["adjacent"] & m["win"], "no_zombie_beside": lambda m: ~m["adjacent"]}
    bayes = {c: float(MA["drop2"][use & f(MA)].float().mean()) for c, f in cases.items()}
    allv = torch.cat([meta["root_visible"][:, None], meta["future_visible"][:, 0]], 1).float()
    hp = allv[:, :, 1512] * 9
    alive = ~meta["future_dead"].cumsum(2).bool().any(1)
    true_frames = torch.cat([ctx.cpu(), fut5[:, 0].cpu()], 1)                        # [R, 4 + 16, 81, 192]
    acts_all = torch.cat([ca.cpu(), fa.cpu()], 1)                                                # [R, 3 + 16]: action t leads frame t -> t+1
    post_h = torch.empty(R, H); prior_hits = torch.zeros(R, H); self_h = torch.empty(R, H)
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
                post_h[i:i + b, k] = health(step(world, post, win, wa, pc[:, c], device, config).to(device)).cpu()
                hist_s, hist_a, hist_c = tf[:, max(0, c + 1 - DW.BLOCKS):c + 1], ac[:, max(0, c + 1 - DW.BLOCKS):c + 1], pc[:, max(0, c + 1 - DW.BLOCKS):c]
                cur = health(tf[:, c]).cpu()
                for _ in range(M):
                    codes, _ = prior.sample(hist_s, hist_a, hist_c, generator=g)
                    prior_hits[i:i + b, k] += (health(step(world, post, win, wa, codes, device, config).to(device)).cpu() < cur - 1.5).float() / M
                iw = min(len(imagined), 5)
                isn = torch.stack(imagined[-DW.BLOCKS:], 1); ian = ac[:, max(0, c + 1 - len(imagined[-DW.BLOCKS:])):c + 1]
                scodes, _ = prior.sample(isn, ian, torch.stack(icodes[-(isn.shape[1] - 1):], 1), generator=g)
                nxt = step(world, post, torch.stack(imagined[-iw:], 1), ac[:, c + 1 - iw:c + 1], scodes, device, config).to(device)
                gen[i:i + b, k] = nxt.half().cpu(); self_h[i:i + b, k] = health(nxt).cpu()
                imagined.append(nxt); icodes.append(scodes)
    dclass = lambda d: torch.bucketize(d, torch.tensor([-1.5, -0.5, 0.5]))
    h_true = torch.stack([health(true_frames[:, 3 + k].float()).cpu() for k in range(H + 1)], 1)          # probe on true frames
    true_c, post_c = dclass(hp[:, 1:] - hp[:, :-1]), dclass(post_h - h_true[:, :-1])
    out = {"samples": M, "bayes": bayes}
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
