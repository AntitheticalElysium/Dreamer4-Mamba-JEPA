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
Usage: check_h16_traj.py <world.pt> ...
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
def features(world, config, data, path, width, device, batch=16):
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
                frames = torch.cat([frames, out[:, -1].float().cpu()[:, None]], 1)[:, -5:]
                acts = torch.cat([acts, cont[:, k - 1:k]], 1)
    mm.flush()
    return torch.from_numpy(np.memmap(path, dtype=np.float16, mode="r", shape=(R, N, H, 3 * width)))


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
    for path in sys.argv[1:]:
        world, st = T.load_world(Path(path), device)
        name = st["name"]
        tag = E.CACHE / f"h16traj_{name}"
        xs = {s: features(world, config, data[s], f"{tag}_{s}.f16", S.D, device) for s in ("fit", "dev")}
        del world; torch.cuda.empty_cache()
        flat = xs["fit"].flatten(0, 2)
        mu = torch.stack([flat[i:i + 65536].float().mean(0) for i in range(0, len(flat), 65536)]).mean(0)
        sd = torch.stack([flat[i:i + 65536].float().std(0) for i in range(0, len(flat), 65536)]).mean(0).clamp(min=1e-3)
        stdz = lambda x: ((x.float() - mu) / sd).half()
        xf, xd = stdz(xs["fit"]), stdz(xs["dev"])
        res, safe = {}, {}
        for arm, traj in (("trajectory", True), ("snapshot", False)):
            runs = []
            for seed in range(3):
                model, best = fit(xf, Pf, xd[A], Pd[A], traj, seed, device)
                s = scores(model, xd[B], traj, device)[oppB]
                runs.append(1 - p16B[torch.arange(len(p16B)), s.argmin(1)])
                log(world=name, arm=arm, seed=seed, devA=round(best, 4), devB=round(float(runs[-1].mean()), 4))
            safe[arm] = torch.stack(runs).mean(0)
            res[arm] = float(safe[arm].mean())
        res["traj_minus_snapshot"] = paired(safe["trajectory"], safe["snapshot"], seeds[B][oppB], draws=1000, seed=20261003)
        out[name] = res
        log(world=name, **res)
        for f in E.CACHE.glob(f"h16traj_{name}_*"):
            f.unlink()
    ws = [k for k in out if k != "references"]
    out["readings"] = {"traj_gain": all(out[w]["traj_minus_snapshot"]["interval"][0] > 0 for w in ws),
                       "traj_reaches_one_future": all(out[w]["trajectory"] >= refs["one_real_future"] - 0.01 for w in ws)}
    (OUT / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
