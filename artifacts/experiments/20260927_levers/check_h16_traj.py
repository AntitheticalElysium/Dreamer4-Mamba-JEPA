"""Decision check (2026-10-03): read step by step, does a deterministic world's imagined TRAJECTORY carry the H16 decision that its
depth-16 snapshot does not? (check_rootaware: a head that reads one imagined state understates what imagination carries -- next to
zombies 0.87-0.91 successor-only vs 0.95 with the root. deepeval's gen16 head reads the depth-16 snapshot only: sealed 0.593-0.599
vs prior 0.586. check_h16_value: one real future per action 0.659 on DEV, oracle 0.740. DreamerV3 / Delta-IRIS / Dedieu et al. read
risk with a per-step termination (continue) head along imagined trajectories.)
deepeval's FIT and DEV roots and imagination (teval convention, recorded continuation). At every imagined step k = 1..16 the world's
hidden state for its prediction of frame k: [mean over tiles, mean over the 3 x 3 around the player, mean over the HUD] (768).
Heads (MLP, features standardized on FIT, 3 head seeds, selection on DEV-A, judged on DEV-B; DEV split by episode seed parity):
  trajectory   per-step hazard logit (step embedding) trained with BCE on the conditional hazard q_k = (P_k - P_{k-1}) /
               (1 - P_{k-1}) weighted by 1 - P_{k-1} (P = 32-key P(dead by k)); P(dead by 16) = 1 - prod_k (1 - hazard_k)
  snapshot     the same MLP on the depth-16 step only, BCE on P(dead by 16)
Judged: expected safe at 16 on DEV-B opp16 roots (argmin over the 17 first actions, seed mean); paired episode-seed-clustered
interval of trajectory - snapshot; references on the same roots: uniform, FIT prior, one_real_future, oracle31 (check_h16_value).
Readings, declared before running (worlds: the 36k teacher worlds, seeds 7 and 8):
  traj_gain                 trajectory - snapshot > 0, interval excluding 0, at both seeds
  traj_reaches_one_future   trajectory >= one_real_future - 0.01 at both seeds
Risk-suite additions (2026-10-03, after the first run started; reported): `--window W` (imagination context, default 5);
continuation / death prediction of the trajectory head on DEV-B: Brier of the per-step cumulative P(dead by k) against the 32-key
P (all roots, branches, k), and AUC of P(dead by 16) against key 0's realized death by 16; per-world JSON (evals/h16traj/<name>.json).
E16 (`--e16`, 2026-10-03; the paths are dworld.py stage-B prior files): imagination samples every step's Delta from the prior and
decodes it (features_e16); the heads are fit as above on one sample. DEV-B is then re-imagined to 16 samples per action and the
predicted P(dead by 16) averaged over the first M = 1 / 4 / 16 samples, from the trajectory heads and from the prior's own end
head (1 - prod_k (1 - P(end_k)), no fitting): e16_h16's quantities.
Usage: check_h16_traj.py <world.pt> ... [--window W]   |   check_h16_traj.py --e16 <prior.pt> ...
Result (2026-10-03, first run, 36k worlds; DEV-B opp16 roots 1,139; references uniform 0.571, prior 0.616, one_real_future
0.679, oracle31 0.766): trajectory / snapshot = 0.645 / 0.629 (s7), 0.646 / 0.629 (s8); trajectory - snapshot +0.017 [+0.007,
+0.026] (s7), +0.017 [+0.005, +0.031] (s8). traj_gain TRUE, traj_reaches_one_future FALSE (-0.033 at both seeds).
Share of the uniform-to-oracle margin: prior 23%, snapshot 30%, trajectory 38%, one real future 55% (check_h16_value: 16 sampled
futures ~90%). Reading the deterministic trajectory step by step helps, but futures that never draw damage stay well below one
faithful sample.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_h16_signal as HS  # noqa: E402
import deepeval as E  # noqa: E402
D, T = E.D, E.T
N, NEAR, H = 17, [21, 22, 23, 30, 31, 32, 39, 40, 41], 16
OUT = Path("artifacts/experiments/20260927_levers/evals/h16traj")


@torch.no_grad()
def features(world, config, data, path, width, device, batch=16, window=5):
    """[R,17,16,3*width] fp16 memmap: the world's hidden state for each imagined step (deepeval.imagine's rollout)."""
    from d4mj.train import autocast_context
    R = len(data["seed"])
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(R, N, H, 3 * width))
    for i in range(0, R, batch):
        frames = data["ctx"][i:i + batch].float().repeat_interleave(N, 0)
        b = len(frames) // N
        acts = torch.cat([data["acts"][i:i + batch].repeat_interleave(N, 0), torch.arange(N).repeat(b)[:, None]], 1)
        cont = data["cont"][i:i + batch].repeat_interleave(N, 0)
        for k in range(1, H + 1):
            with autocast_context(config):
                out, h, _ = world(frames.to(device), acts[:, -frames.shape[1]:].to(device))
            hk = h[:, -1].float()
            mm[i:i + b, :, k - 1] = torch.cat([hk.mean(1), hk[:, NEAR].mean(1), hk[:, 63:81].mean(1)], -1).view(b, N, -1).half().cpu().numpy()
            if k < H:
                frames = torch.cat([frames, out[:, -1].float().cpu()[:, None]], 1)[:, -window:]
                acts = torch.cat([acts, cont[:, k - 1:k]], 1)
    mm.flush()
    return torch.from_numpy(np.memmap(path, dtype=np.float16, mode="r", shape=(R, N, H, 3 * width)))


@torch.no_grad()
def features_e16(world, post, prior, config, data, path, width, device, generator, batch=16, window=5):
    """E16 (dworld.py): as features(), but every step's Delta is SAMPLED by the stage-B prior given the true context (posterior
    codes for the 3 context transitions) and the imagined history, then decoded. Also P(end) per step from the prior's end head.
    -> features [R,17,16,3*width] (memmap), p_end [R,17,16]"""
    import dworld as DW
    from d4mj.train import autocast_context
    R = len(data["seed"])
    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(R, N, H, 3 * width))
    pend = torch.empty(R, N, H)
    for i in range(0, R, batch):
        ctx = data["ctx"][i:i + batch].float().to(device)
        b = len(ctx)
        past = data["acts"][i:i + batch].to(device)
        with torch.no_grad():
            pc = post.quantizer(post.encode(ctx[:, :-1].flatten(0, 1), past.flatten(), ctx[:, 1:].flatten(0, 1)))[1].view(b, 3, DW.K)
        frames = ctx.repeat_interleave(N, 0)
        acts = torch.cat([past.repeat_interleave(N, 0), torch.arange(N, device=device).repeat(b)[:, None]], 1)
        codes = pc.repeat_interleave(N, 0)
        cont = data["cont"][i:i + batch].to(device).repeat_interleave(N, 0)
        for k in range(1, H + 1):
            sampled, p_end = prior.sample(frames[:, -DW.BLOCKS:], acts[:, -DW.BLOCKS:], codes[:, -(DW.BLOCKS - 1):], generator=generator)
            w = min(frames.shape[1], window)
            delta = torch.zeros(len(frames), w, 81, DW.S.D, device=device)
            delta[:, -1] = post.condition(post.quantizer.embed(sampled))
            world.delta = delta
            with autocast_context(config):
                out, h, _ = world(frames[:, -w:], acts[:, -w:])
            world.delta = None
            hk = h[:, -1].float()
            mm[i:i + b, :, k - 1] = torch.cat([hk.mean(1), hk[:, NEAR].mean(1), hk[:, 63:81].mean(1)], -1).view(b, N, -1).half().cpu().numpy()
            pend[i:i + b, :, k - 1] = p_end.view(b, N).cpu()
            frames = torch.cat([frames, out[:, -1].float()[:, None]], 1)
            codes = torch.cat([codes, sampled[:, None]], 1)
            if k < H:
                acts = torch.cat([acts, cont[:, k - 1:k]], 1)
    mm.flush()
    return torch.from_numpy(np.memmap(path, dtype=np.float16, mode="r", shape=(R, N, H, 3 * width))), pend


class Hazard(nn.Module):
    def __init__(self, d, steps):
        super().__init__()
        self.step = nn.Parameter(torch.zeros(steps, 64))
        self.net = nn.Sequential(nn.Linear(d + 64, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, x):                                                            # [B,steps,d] -> logits [B,steps]
        return self.net(torch.cat([x, self.step.expand(len(x), -1, -1)], -1))[..., 0]


def p16(model, x, traj):
    """P(dead by 16) per (root, branch) from the model: x [M,16,d] standardized."""
    z = model(x if traj else x[:, -1:])
    return 1 - torch.exp(-F.softplus(z).sum(1)) if traj else torch.sigmoid(z[:, 0])


def fit(x, P, xa, Pa, traj, seed, device, steps=4000):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = Hazard(x.shape[-1], H if traj else 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    prev = F.pad(P, (1, 0))[..., :-1]                                                # P_{k-1}, [R,17,16]
    q = ((P - prev) / (1 - prev).clamp(min=1e-6)).clamp(0, 1)
    w = (1 - prev).clamp(min=0)
    flat = lambda t: t.flatten(0, 1)
    xf, qf, wf, pf = flat(x), flat(q), flat(w), flat(P[..., -1])
    best, state = -1.0, None
    for step in range(steps):
        idx = torch.randint(len(xf), (512,), generator=g)
        xb = xf[idx].float().to(device)
        if traj:
            z = model(xb)
            loss = (wf[idx].to(device) * F.binary_cross_entropy_with_logits(z, qf[idx].to(device), reduction="none")).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(model(xb[:, -1:])[:, 0], pf[idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 200 == 0:
            v = judge(model, xa, Pa, traj, device)
            if v > best:
                best, state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(state)
    return model.eval(), best


@torch.no_grad()
def cumulative(model, x, device):
    """trajectory head: P(dead by k) for k = 1..16, [R,17,16]"""
    return torch.cat([(1 - torch.exp(-F.softplus(model(x[i:i + 256].flatten(0, 1).float().to(device))).cumsum(1))).view(-1, N, H).cpu()
                      for i in range(0, len(x), 256)])


def auc(score, label):
    order = score.argsort()
    ranks = torch.empty_like(order, dtype=torch.float); ranks[order] = torch.arange(1, len(score) + 1, dtype=torch.float)
    pos = label.bool(); n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


@torch.no_grad()
def scores(model, x, traj, device):
    return torch.cat([p16(model, x[i:i + 256].flatten(0, 1).float().to(device), traj).view(-1, N).cpu()
                      for i in range(0, len(x), 256)])


def judge(model, x, P, traj, device):
    s = scores(model, x, traj, device)
    opp = P[..., -1].amax(1) > P[..., -1].amin(1)
    pick = s.argmin(1)
    return float(1 - P[torch.arange(len(P)), pick, -1][opp].mean())


def main():
    from d4mj.config import config_from_dict
    from ladder import paired
    import spatial as S
    device = torch.device("cuda")
    args = sys.argv[1:]
    window = int(args[args.index("--window") + 1]) if "--window" in args else 5
    e16 = "--e16" in args                       # E16: the paths are stage-B prior files (dworld.py); imagination samples Delta
    paths = [a for i, a in enumerate(args) if a not in ("--window", "--e16") and (i == 0 or args[i - 1] != "--window")]
    log = lambda **kw: print(json.dumps(kw), flush=True)
    data = E.token_cache(device, log)
    Pall, D0, split = HS.load(full=True)
    for i, name in enumerate(("fit", "dev")):
        p = Pall[split == i][..., [0, 3, 15]]
        ref = torch.stack([data[name][f"p{k}"] for k in E.DEPTHS], -1)
        if p.shape != ref.shape or not torch.allclose(p, ref.float()):
            raise SystemExit(f"{name}: deeppanel rows do not match deepeval's cache order")
    Pf, Pd, Dd = Pall[split == 0], Pall[split == 1], D0[split == 1][..., 2]
    seeds = data["dev"]["seed"]
    A, B = seeds % 2 == 0, seeds % 2 == 1
    oppB = Pd[B][..., -1].amax(1) > Pd[B][..., -1].amin(1)
    p16B, dB = Pd[B][..., -1][oppB], Dd[B][oppB]
    fit_opp = Pf[..., -1].amax(1) > Pf[..., -1].amin(1)
    prior = int(Pf[fit_opp][..., -1].mean(0).argmin())
    rest = (32 * p16B - dB) / 31
    alive0 = dB == 0
    sel = torch.where(alive0.any(1, keepdim=True), alive0, torch.ones_like(alive0)).float()
    refs = {"roots_devB_opp16": int(oppB.sum()), "uniform": float(1 - p16B.mean()), "prior": float(1 - p16B[:, prior].mean()),
            "one_real_future": float(1 - ((sel / sel.sum(1, keepdim=True)) * rest).sum(1).mean()),
            "oracle31": float(1 - dB[torch.arange(len(dB)), rest.argmin(1)].mean())}
    log(references=refs)
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"references": refs}
    for path in paths:
        if e16:
            import dworld as DW
            st = torch.load(path, map_location="cpu", weights_only=False)
            world, post, _ = DW.load_a(Path(st["stage_a"]), device)
            dyn = DW.Prior().to(device); dyn.load_state_dict(st["prior"]); dyn.eval()
            gen = torch.Generator(device=device).manual_seed(20261003)
            name = st["name"]
            tag = E.CACHE / f"h16traj_{name}"
            xs, pend = {}, {}
            for sp in ("fit", "dev"):
                xs[sp], pend[sp] = features_e16(world, post, dyn, config, data[sp], f"{tag}_{sp}.f16", S.D, device, gen, window=window)
        else:
            world, st = T.load_world(Path(path), device)
            name = st["name"]
            tag = E.CACHE / f"h16traj_{name}"
            xs = {s: features(world, config, data[s], f"{tag}_{s}.f16", S.D, device, window=window) for s in ("fit", "dev")}
            del world; torch.cuda.empty_cache()
        flat = xs["fit"].flatten(0, 2)
        mu = torch.stack([flat[i:i + 65536].float().mean(0) for i in range(0, len(flat), 65536)]).mean(0)
        sd = torch.stack([flat[i:i + 65536].float().std(0) for i in range(0, len(flat), 65536)]).mean(0).clamp(min=1e-3)
        stdz = lambda x: ((x.float() - mu) / sd).half()
        xf, xd = stdz(xs["fit"]), stdz(xs["dev"])
        res, safe, traj_models = {}, {}, []
        for arm, traj in (("trajectory", True), ("snapshot", False)):
            runs, cont = [], []
            for seed in range(3):
                model, best = fit(xf, Pf, xd[A], Pd[A], traj, seed, device)
                if traj:
                    traj_models.append(model)
                s = scores(model, xd[B], traj, device)[oppB]
                runs.append(1 - p16B[torch.arange(len(p16B)), s.argmin(1)])
                log(world=name, arm=arm, seed=seed, devA=round(best, 4), devB=round(float(runs[-1].mean()), 4))
                if traj:                                                             # continuation / death prediction, DEV-B
                    c = cumulative(model, xd[B], device)
                    cont.append({"brier_all_k": float(((c - Pd[B]) ** 2).mean()),
                                 "auc_dead16_key0": auc(c[..., -1].flatten(), Dd[B].flatten())})
            if cont:
                res["continuation"] = {k: sum(x[k] for x in cont) / len(cont) for k in cont[0]}
            safe[arm] = torch.stack(runs).mean(0)
            res[arm] = float(safe[arm].mean())
        res["traj_minus_snapshot"] = paired(safe["trajectory"], safe["snapshot"], seeds[B][oppB], draws=1000, seed=20261003)
        if e16:                                     # M sampled futures per action on DEV-B: average the predicted P(dead by 16)
            devB = {k: data["dev"][k][B] for k in ("ctx", "acts", "cont", "seed")}
            judge_risk = lambda r: float((1 - p16B[torch.arange(len(p16B)), r[oppB].argmin(1)]).mean())
            r_traj, r_end = [], []
            for m in range(16):
                if m == 0:
                    fx, pe = xd[B], pend["dev"][B]
                else:
                    fx, pe = features_e16(world, post, dyn, config, devB, f"{tag}_devB_m.f16", S.D, device, gen, window=window)
                    fx = stdz(fx)
                r_traj.append(torch.stack([scores(mdl, fx, True, device) for mdl in traj_models]).mean(0))
                r_end.append(1 - torch.prod(1 - pe, -1))
            res["e16_samples"] = {f"M{M}": {"trajectory_head": judge_risk(torch.stack(r_traj[:M]).mean(0)),
                                            "end_head": judge_risk(torch.stack(r_end[:M]).mean(0))} for M in (1, 4, 16)}
            del world, post, dyn; torch.cuda.empty_cache()
        res["window"] = window
        out[name] = res
        (OUT / f"{name}{'' if window == 5 else f'__w{window}'}.json").write_text(json.dumps(res | {"references": refs}, indent=2) + "\n")
        log(world=name, **res)
        for f in E.CACHE.glob(f"h16traj_{name}_*"):
            f.unlink()
    ws = [k for k in out if k != "references"]
    out["readings"] = {"traj_gain": all(out[w]["traj_minus_snapshot"]["interval"][0] > 0 for w in ws),
                       "traj_reaches_one_future": all(out[w]["trajectory"] >= refs["one_real_future"] - 0.01 for w in ws)}
    target = OUT / ("result.json" if ws == ["corrt_raw_teacher_s7_u36000", "corrt_raw_teacher_s8_u36000"] and window == 5 and not e16
                    else f"result_{ws[0]}{'' if window == 5 else f'__w{window}'}.json")      # the first run keeps result.json
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
