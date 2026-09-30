"""E6. Why option (b) does not beat (a) at matched compute -- exactly.

Measured (E1e/E1f): err(long world, full history) = err(long world, own 3-frame window) x (1 - memory benefit).
L4to64b24 at positions 4-7: 0.72 x (1 - 0.18) = 0.60, against the windowed L4b512's 0.52. Memory helps; the long-
trained world is worse at the short-context core (0.72 vs 0.52). Candidate causes and the arm that decides each,
everything else as E1 (frozen Raw latents, fresh canonical Mamba, joint optimizer settings, 10,000 updates,
~1,500 transitions per update, init seed 7, sampler seed 11):

  diversity   L64b24 sees 24 episodes' 64-frame segments per update; L4b512 sees 512 independent windows.
              L4chop: EXACTLY L64b24's sampled segments (same sampler, same 24 x 64 frames), cut into 21 consecutive
              4-frame windows each (504 windows, 1,512 transitions), each scanned from a zero state. Same data as
              L64b24, no long context. If L4chop ~ L64b24's window -> the data's correlation is the cause; if
              L4chop ~ L4b512 -> training on long sequences is.
  regime      A long-trained model spends 3/63 of its loss on positions 1-3. Lalt: Dreamer 4's alternating batch
              lengths -- even updates L4 x 512, odd updates L64 x 24 (same per-update transitions).
  ceiling     How much of the next latent is predictable from older history at all, before any training dynamics:
              closed-form ridge on DEV/FINAL 128-frame windows, target dz = z_{t+1} - z_t, features
              short = [z_{t-2}, z_{t-1}, z_t, onehot(a_t), onehot(a_{t-1}), onehot(a_{t-2})], long = short +
              [z_{t-3} ... z_{t-15}] (and + means of z over t-16..t-63). R^2 on held-out windows.
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import context_length as C  # noqa: E402

OUT = HERE / "bdiag.json"


def train_arm(arm, config, episodes, device, log):
    from d4mj.train import autocast_context, learning_rate, optimizer, optimizer_step
    world = C.new_world(config, device)
    opt = optimizer([world], config.joint, exclude_vectors=True)
    params = [p for g in opt.param_groups for p in g["params"]]
    train = [e for e in episodes if e.split == "train"]
    long = C.Windows(train, 64, seed=11)
    short = C.Windows(train, 4, seed=12)
    history, started = [], time.time()
    for update in range(C.UPDATES):
        if arm == "L4chop":
            z, a = long.sample(24)                                        # [24,64,1,D], [24,63]
            idx = torch.arange(0, 61, 3)                                  # 21 windows: frames s..s+3
            z = torch.stack([z[:, s:s + 4] for s in idx.tolist()], 1).flatten(0, 1)
            a = torch.stack([a[:, s:s + 3] for s in idx.tolist()], 1).flatten(0, 1)
        elif arm == "Lalt":
            z, a = short.sample(512) if update % 2 == 0 else long.sample(24)
        z, a = z.to(device), a.to(device)
        with autocast_context(config):
            pred = world.teacher(z, a).predicted
        loss = (pred.float() - z[:, 1:].float()).square().mean()
        norm = optimizer_step(opt, loss, params, learning_rate=learning_rate(config, update),
                              grad_clip=config.joint.grad_clip, strict=True, zero_grad=True)
        if (update + 1) % 1000 == 0:
            row = {"update": update + 1, "loss": float(loss.detach()), "gradient_norm": float(norm),
                   "seconds": round(time.time() - started, 1)}
            history.append(row)
            log(stage="train", arm=arm, **row)
    return world.eval(), history


@torch.no_grad()
def ceiling(episodes, device):
    held = [e for e in episodes if e.split in ("dev", "final")]
    w = C.Windows(held, 128, seed=21)
    z, a = w.sample(512)
    z = z[:, :, 0].double()                                               # [n,128,D]
    oh = F.one_hot(a, 17).double()                                        # [n,127,17]
    feats_s, feats_l, feats_ll, targets, groups = [], [], [], [], []
    for t in range(64, 127):
        s = torch.cat([z[:, t - 2], z[:, t - 1], z[:, t], oh[:, t], oh[:, t - 1], oh[:, t - 2]], 1)
        l = torch.cat([s] + [z[:, t - k] for k in range(3, 16)], 1)
        ll = torch.cat([l, z[:, t - 63:t - 15].mean(1)], 1)
        feats_s.append(s); feats_l.append(l); feats_ll.append(ll)
        targets.append(z[:, t + 1] - z[:, t]); groups.append(torch.arange(len(z)))
    S, L, LL, Y, G = (torch.cat(v) for v in (feats_s, feats_l, feats_ll, targets, groups))
    tr, te = G < int(0.7 * len(z)), G >= int(0.7 * len(z))
    out = {}
    for name, X in (("short_3", S), ("long_16", L), ("long_16_plus_mean_64", LL)):
        mu, sd = X[tr].mean(0), X[tr].std(0).clamp_min(1e-6)
        Xs = torch.cat([(X - mu) / sd, torch.ones(len(X), 1, dtype=X.dtype)], 1).to(device)
        Yd = Y.to(device)
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
            trd, ted = tr.to(device), te.to(device)
            wgt = torch.linalg.solve(Xs[trd].T @ Xs[trd] + lam * int(tr.sum()) * torch.eye(Xs.shape[1], dtype=Xs.dtype, device=device),
                                     Xs[trd].T @ Yd[trd])
            r2 = float(1 - ((Xs[ted] @ wgt - Yd[ted]) ** 2).sum() / ((Yd[ted] - Yd[trd].mean(0)) ** 2).sum())
            if best is None or r2 > best:
                best = r2
        out[name] = best
    return out


def main():
    device = torch.device("cuda")
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds_total": round(time.time() - started, 1)}), flush=True)
    config, episodes = C.load(device)
    result = json.loads(OUT.read_text()) if OUT.exists() else {}
    if "ceiling" not in result:
        result["ceiling"] = ceiling(episodes, device)
        log(stage="ceiling", **result["ceiling"])
        OUT.write_text(json.dumps(result, indent=2) + "\n")
    for arm in ("L4chop", "Lalt"):
        if arm in result:
            continue
        path = C.OUT / f"{arm}.pt"
        if path.exists():
            world = C.new_world(config, device)
            world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"]); world.eval()
            history = []
        else:
            world, history = train_arm(arm, config, episodes, device, log)
            torch.save({"world": world.state_dict(), "history": history, "arm": arm}, path)
        result[arm] = {"history": history, **C.evaluate(world, config, episodes, device)}
        log(stage="eval", arm=arm, **{k: v for k, v in result[arm].items() if k != "history"})
        OUT.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
