"""The RIGHT test of whether a WM is usable for imagination-RL: ACTION SENSITIVITY. For a given state,
does the WM predict DIFFERENT next-latents for different actions? If the prediction is ~action-invariant,
the policy cannot learn anything in imagination (all actions look identical) -- independent of 'beats copy'.
Metric: std of the next-latent prediction ACROSS the 17 actions, relative to the real frame-to-frame change.
ratio << 1  => WM barely reacts to actions  => imagination is useless for a policy.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "third_party", "Drama"))
import crafter
from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from train_online_jepa import enc_one
dev = "cuda"


def load(tag):
    ck = torch.load(os.path.join(REPO, "ckpt", f"jepa_wm_{tag}.pt"), map_location=dev, weights_only=False); D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    return enc, model, D


@torch.no_grad()
def collect(enc, n=8, maxs=200):
    trajs = []; rng = np.random.default_rng(0)
    for sd in range(5000, 5000 + n):
        env = crafter.Env(seed=sd); o = env.reset(); es, as_ = [], []
        for _ in range(maxs):
            z, _ = enc_one(enc, o, dev); a = int(rng.integers(17))
            es.append(z.squeeze(0)); as_.append(a); o, r, done, _ = env.step(a)
            if done: break
        trajs.append((torch.stack(es), torch.tensor(as_, device=dev)))
    return trajs


@torch.no_grad()
def action_sensitivity(model, trajs, D, C=8, nwin=150):
    """For each window: backbone on the C-step context, vary the LAST action over all 17, measure the
    spread of predicted next-latent across actions vs the real frame change."""
    rng = np.random.default_rng(1); spreads, changes = [], []
    for _ in range(nwin):
        es, as_ = trajs[rng.integers(len(trajs))]
        if len(as_) < C + 1: continue
        s = int(rng.integers(0, len(as_) - C - 1))
        e_ctx = es[s:s + C].unsqueeze(0).expand(17, C, D).contiguous()      # (17, C, D) same states
        a_ctx = as_[s:s + C].unsqueeze(0).expand(17, C).contiguous().clone()
        a_ctx[:, -1] = torch.arange(17, device=dev)                          # vary the last action
        preds, _ = model(e_ctx, a_ctx)                                       # (K,17,C,D)
        nextpred = preds.float().mean(0)[:, -1]                              # (17, D) next-latent per action
        spread = nextpred.std(dim=0).mean().item()                          # spread across actions
        change = (es[s + C] - es[s + C - 1]).abs().mean().item()            # real frame-to-frame change
        spreads.append(spread); changes.append(change)
    return float(np.mean(spreads)), float(np.mean(changes))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=["online_ls_v1"]); a = ap.parse_args()
    for tag in a.tags:
        enc, model, D = load(tag)
        spread, change = action_sensitivity(model, collect(enc), D)
        print(f"{tag:22s} | action-spread {spread:.4f} | real frame-change {change:.4f} | "
              f"ratio {spread/ (change+1e-9):.3f}  (<<1 => WM ignores actions => imagination useless)")
        del enc, model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
