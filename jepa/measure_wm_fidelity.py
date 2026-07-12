"""Measure imagination FIDELITY of the best WM (ls_v1): does the imagined latent trajectory track
the REAL one? Roll the WM H steps feeding its own predictions, using the REAL actions the policy took,
and compare each imagined latent to the real latent. If it diverges fast, imagination is unreliable and
the WM -- not policy training -- is the ceiling. Baseline: 'copy' (predict no change) divergence.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "third_party", "Drama"))
import crafter
from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from agents import ActorCriticAgent
from train_agent import shim_config
from train_online_jepa import enc_one
dev = "cuda"


def load(wm):
    ck = torch.load(os.path.join(REPO, "ckpt", wm), map_location=dev, weights_only=False); D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    return enc, model


@torch.no_grad()
def collect(enc, model, n=8, maxs=300):
    """RANDOM-action trajectories -> WM's own encoder embeddings + actions (policy-independent WM test)."""
    trajs = []; rng = np.random.default_rng(0)
    for sd in range(5000, 5000 + n):
        env = crafter.Env(seed=sd); o = env.reset(); es, as_ = [], []
        for _ in range(maxs):
            z, _ = enc_one(enc, o, dev); a = int(rng.integers(17))
            es.append(z.squeeze(0)); as_.append(a)
            o, r, done, _ = env.step(a)
            if done: break
        trajs.append((torch.stack(es), torch.tensor(as_, device=dev)))
    return trajs


@torch.no_grad()
def fidelity(model, trajs, C=8, H=16, nwin=200):
    rng = np.random.default_rng(0); div = np.zeros(H); cpy = np.zeros(H); cnt = 0
    for _ in range(nwin):
        es, as_ = trajs[rng.integers(len(trajs))]
        if len(as_) < C + H + 1: continue
        s = int(rng.integers(0, len(as_) - C - H))
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        for t in range(C - 1):                                    # prime context
            _, h, states = model.step(es[s + t][None, None], as_[s + t][None], states)
        z_cur = es[s + C - 1][None, None]                         # start from last real latent
        for i in range(H):
            preds, h, states = model.step(z_cur, as_[s + C - 1 + i][None], states)
            zhat = preds.float().mean(0)                          # imagined next latent (feed own prediction)
            real = es[s + C + i][None, None]
            div[i] += (F.l1_loss(zhat, real) / (real.abs().mean() + 1e-9)).item()
            cpy[i] += (F.l1_loss(z_cur, real) / (real.abs().mean() + 1e-9)).item()   # copy baseline
            z_cur = zhat
        cnt += 1
    return div / cnt, cpy / cnt


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=["online_ls_v1"]); a = ap.parse_args()
    for tag in a.tags:
        enc, model = load(f"jepa_wm_{tag}.pt")
        div, cpy = fidelity(model, collect(enc, model))
        beats = [i + 1 for i in range(16) if div[i] < cpy[i] - 0.01]   # meaningfully beats copy
        # gain = how much better than copy, averaged over horizon (positive = WM adds signal)
        gain = float(np.mean(cpy - div))
        print(f"\n=== {tag} ===")
        print("  step:  " + " ".join(f"{i+1:5d}" for i in range(0, 16, 3)))
        print("  WM  :  " + " ".join(f"{div[i]:5.2f}" for i in range(0, 16, 3)))
        print("  copy:  " + " ".join(f"{cpy[i]:5.2f}" for i in range(0, 16, 3)))
        print(f"  mean gain over copy: {gain:+.3f}  | meaningfully beats copy at steps: {beats or 'NONE'}")
        del enc, model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
