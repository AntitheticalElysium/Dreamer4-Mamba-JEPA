"""D10. Every world rolled on the diagnostic futures (D9): imagined, teacher-forced, and one-step all-action.

Worlds (frozen, nothing trained):
  H2   canonical Raw H2 LeWMWorld (Mamba), raw z           | its own world, bridge-trained, depth 2
  Z    interface Mamba world, raw z                        | interface.py recipe, alias-free depth 1-2
  U    interface Mamba world, u = PCA-192 of the 4x4 grid  |
  W    interface Mamba world, w = u / std                  |
  sZ   spatial.py block-causal transformer, layer-normed z | trained with a depth-2 suffix
  T    spatial.py block-causal transformer, 81 layer-normed tokens
Per root, with the policy's own 16 actions:
  gen    imagined rollout, depth 1..16 (Mamba: repeated `advance`; transformer: a sliding window of the
         last 5 frames, 4 at depth 1, the window the transformers were trained on)
  tf     teacher-forced one-step prediction at every depth (the true frames of sample 0 fed back)
  true   the encoded true states of samples 0-4 (sample 0 = the factual trajectory)
  one    depth-1 prediction for all 17 actions; `one_true` the 4-key true successors; Mamba `one_h` = h(a)
T is stored in a PCA-1024 of its flattened tokens (fitted on the true futures of the first 128 roots);
its full-space errors are computed on the fly: per depth and per token.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER))
import interface as I  # noqa: E402

STORE = ROOT / "artifacts/eda/diagnosis_futures_v1"
OUT = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
EDA = ROOT / "artifacts/eda"
H, N, KS = 16, 17, 5
MAMBA = {"H2": "Z", "Z": "Z", "U": "U", "W": "W"}
SPATIAL = {"sZ": "z", "T": "tokens"}
PCA_T = 1024


def load_roots():
    return [r for f in sorted(STORE.glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]


@torch.no_grad()
def encode(encoder, frames, device, batch=128):
    """uint8 [n, 63, 63, 3] -> raw z [n,192], grid [n,3072] (export's pooling), LN tokens [n,81,192] fp16."""
    zs, gs, ts = [], [], []
    for i in range(0, len(frames), batch):
        z, _, tokens, _, _ = encoder._hidden(frames[i:i + batch, None].to(device))
        pooled = F.adaptive_avg_pool2d(tokens.transpose(1, 2).reshape(len(tokens), 192, 9, 9), 4)
        zs.append(z.float()); gs.append(pooled.flatten(2).transpose(1, 2).flatten(1).float())
        ts.append(F.layer_norm(tokens.float(), (192,)).half())
    return torch.cat(zs), torch.cat(gs), torch.cat(ts)


def states(pool, z, grid, tokens):
    return {"H2": z, "Z": z, "U": I.state_of("U", pool["pca"], z, grid), "W": I.state_of("W", pool["pca"], z, grid),
            "sZ": F.layer_norm(z, (192,)), "T": tokens.float()}


def load_worlds(device):
    from d4mj.agent import Heads  # noqa: F401
    from d4mj.experiments import _load_bridge_parent
    import spatial as S
    encoder, config = I.load_bridge()
    worlds = {}
    bundle, _, _ = _load_bridge_parent(I.CHECKPOINT)
    bundle.world.eval()
    worlds["H2"] = bundle
    for name, path in (("Z", EDA / "interface_worlds_v1/Z.pt"), ("U", EDA / "interface_worlds_v1/U.pt"),
                       ("W", EDA / "interface_worlds_white/W.pt")):
        b = I.world_bundle(config, encoder, device)
        b.world.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["world"])
        b.world.eval()
        worlds[name] = b
    for name, arm in (("sZ", "Z"), ("T", "T")):
        w = S.World(1 if arm == "Z" else 81, False).to(device)
        state = torch.load(EDA / f"spatial_worlds_v1/{arm}.pt", map_location="cpu", weights_only=False)["world"]
        w.load_state_dict(state, strict=True)
        worlds[name] = w.eval()
    return encoder, config, worlds


@torch.no_grad()
def mamba(bundle, config, ctx, ctx_a, fut_a, true0, device):
    """ctx [B,4,D], ctx_a [B,3], fut_a [B,16], true0 [B,16,D] -> gen, tf, one, one_h."""
    from d4mj.train import autocast_context
    with autocast_context(config):
        st = bundle.world.teacher(ctx[:, :, None].to(device), ctx_a.to(device)).state
        cur, gen = st, []
        for k in range(H):
            cur, _ = bundle.advance(cur, fut_a[:, k:k + 1].to(device))
            gen.append(cur.latent[:, 0, 0].float())
        seq = torch.cat([ctx, true0], 1)[:, :, None].to(device)
        tf = bundle.world.teacher(seq, torch.cat([ctx_a, fut_a], 1).to(device)).predicted[:, 3:3 + H, 0].float()
        fan = bundle.repeat_state(st, N)
        adv, _ = bundle.advance(fan, torch.arange(N, device=device).repeat(len(ctx))[:, None])
    B = len(ctx)
    return (torch.stack(gen, 1).cpu(), tf.cpu(), adv.latent[:, 0, 0].float().view(B, N, -1).cpu(),
            adv.history[:, 0].float().view(B, N, -1).cpu())


@torch.no_grad()
def spatial(world, config, ctx, ctx_a, fut_a, true0, device):
    """ctx [B,4,n,D] (n=1 or 81), true0 [B,16,n,D]."""
    from d4mj.train import autocast_context

    def step(frames, acts):                                  # frames [B,t,n,D], acts [B,t] -> next [B,n,D]
        return world(frames.to(device), acts.to(device))[0][:, -1].float()

    gen_frames, tf_out = [ctx[:, i] for i in range(4)], []
    true_frames = [ctx[:, i] for i in range(4)]
    acts = [ctx_a[:, i] for i in range(3)]
    gen = []
    with autocast_context(config):
        for k in range(H):
            width = 4 if k == 0 else 5
            a = torch.stack(acts[-(width - 1):] + [fut_a[:, k]], 1)
            g = step(torch.stack(gen_frames[-width:], 1), a).cpu()
            t = step(torch.stack(true_frames[-width:], 1), a).cpu()
            gen.append(g); tf_out.append(t)
            gen_frames.append(g.to(ctx.dtype)); true_frames.append(true0[:, k]); acts.append(fut_a[:, k])
        B = len(ctx)
        s = ctx.repeat_interleave(N, 0)
        a = torch.cat([ctx_a.repeat_interleave(N, 0), torch.arange(N).repeat(B)[:, None]], 1)
        one = step(s, a).cpu().view(B, N, *ctx.shape[2:])
    return torch.stack(gen, 1), torch.stack(tf_out, 1), one


def main(limit=None, batch=16):
    device = torch.device("cuda")
    sys.path.insert(0, str(LADDER))
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    encoder, config, worlds = load_worlds(device)
    roots = load_roots()[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    # PCA basis for T from the true futures of the first 128 roots
    sample = torch.cat([r["future_frames"].flatten(0, 1) for r in roots[:128]])
    _, _, tok = encode(encoder, sample, device)
    X = tok.flatten(1).float().to(device)
    t_mean = X.mean(0)
    _, sv, V = torch.svd_lowrank(X - t_mean, q=PCA_T + 64, niter=4)
    t_basis = V[:, :PCA_T].cpu()
    explained = float((sv[:PCA_T] ** 2).sum() / ((X - t_mean) ** 2).sum())
    t_mean = t_mean.cpu()
    del X, tok, sample
    print(json.dumps({"roots": len(roots), "T_pca_explained": explained}), flush=True)
    proj = lambda x: (x.flatten(-2).float() - t_mean) @ t_basis
    out = {w: {k: [] for k in ("root", "gen", "tf", "true", "one", "one_true", "one_h")} for w in list(MAMBA) + list(SPATIAL)}
    terr = {k: [] for k in ("gen_err", "tf_err", "gen_tf", "noise", "bias", "persist", "one_err_tok", "gen_err_tok")}
    for i in range(0, len(roots), batch):
        rb = roots[i:i + batch]
        B = len(rb)
        ctx_f = torch.stack([r["context"] for r in rb]).flatten(0, 1)
        fut_f = torch.stack([r["future_frames"] for r in rb]).flatten(0, 2)          # [B*5*16]
        one_f = torch.stack([r["onestep_frames"] for r in rb]).flatten(0, 2)         # [B*4*17]
        enc = {k: encode(encoder, f, device) for k, f in (("ctx", ctx_f), ("fut", fut_f), ("one", one_f))}
        ctx_a = torch.stack([r["context_actions"] for r in rb])
        fut_a = torch.stack([r["future_actions"] for r in rb])
        st = {k: states(pool, *[x.cpu() for x in v]) for k, v in enc.items()}
        for w in list(MAMBA) + list(SPATIAL):
            c = st["ctx"][w].view(B, 4, *st["ctx"][w].shape[1:])
            tr = st["fut"][w].view(B, KS, H, *st["fut"][w].shape[1:])
            ot = st["one"][w].view(B, 4, N, *st["one"][w].shape[1:])
            if w in MAMBA:
                gen, tf, one, one_h = mamba(worlds[w], config, c, ctx_a, fut_a, tr[:, 0], device)
                out[w]["one_h"].append(one_h.half())
            else:
                cs = c[:, :, None] if w == "sZ" else c
                t0 = tr[:, 0][:, :, None] if w == "sZ" else tr[:, 0]
                gen, tf, one = spatial(worlds[w], config, cs, ctx_a, fut_a, t0, device)
                if w == "sZ":
                    gen, tf, one = gen[:, :, 0], tf[:, :, 0], one[:, :, 0]
            if w == "T":
                truef = tr.float()
                mean = truef.mean(1)
                terr["gen_err"].append(((gen - truef[:, 0]) ** 2).sum((-1, -2)))
                terr["tf_err"].append(((tf - truef[:, 0]) ** 2).sum((-1, -2)))
                terr["gen_tf"].append(((gen - tf) ** 2).sum((-1, -2)))
                terr["noise"].append(((truef - mean[:, None]) ** 2).sum((-1, -2)).sum(1) / (KS - 1))
                terr["bias"].append(((gen - mean) ** 2).sum((-1, -2)))
                terr["persist"].append(((c[:, -1:].float() - truef[:, 0]) ** 2).sum((-1, -2)))
                terr["gen_err_tok"].append(((gen - truef[:, 0]) ** 2).sum(-1).half())
                terr["one_err_tok"].append(((one[:, None] - ot.float()) ** 2).sum(-1).mean(1).half())
                c, tr, ot, gen, tf, one = proj(c), proj(tr), proj(ot), proj(gen), proj(tf), proj(one)
            out[w]["root"].append(c[:, -1].float()); out[w]["gen"].append(gen.float()); out[w]["tf"].append(tf.float())
            out[w]["true"].append(tr.float()); out[w]["one"].append(one.float()); out[w]["one_true"].append(ot.float())
        if (i // batch) % 10 == 0:
            print(json.dumps({"done": i + B, "of": len(roots)}), flush=True)
    for w, d in out.items():
        torch.save({k: torch.cat(v) for k, v in d.items() if v}, OUT / f"{w}.pt")
    torch.save({k: torch.cat(v) for k, v in terr.items()} | {"t_mean": t_mean, "t_basis": t_basis, "explained": explained},
               OUT / "T_fullspace.pt")
    meta = {k: torch.stack([r[k] for r in roots]) for k in ("root_visible", "root_hidden", "onestep_visible", "onestep_hidden",
                                                            "onestep_dead", "future_actions", "future_visible",
                                                            "future_hidden", "future_dead", "context_actions")}
    meta["seed"] = torch.tensor([r["seed"] for r in roots])
    torch.save(meta, OUT / "meta.pt")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
