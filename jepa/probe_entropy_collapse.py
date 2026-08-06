"""Fast, controlled root-cause probe for the collapse. Hypothesis: many policy-gradient updates
at the weak DRAMA entropy_coef (3e-4) collapse the policy's entropy. Isolate it from the expensive
online loop: FREEZE a good WM, precompute context embeddings, then run PURE imagination training at
several entropy_coefs and watch policy entropy vs #updates. If 3e-4 collapses and higher coefs hold,
entropy_coef is the lever (fix). Minutes, not hours.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "third_party", "Drama"))
import crafter
from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from agents import ActorCriticAgent
from train_agent import shim_config, imagine
dev = "cuda"


@torch.no_grad()
def collect_frames(enc, n_eps=10, max_steps=300):
    """Random-policy frames -> encoded z (N,D), actions (N,), episode-start flags."""
    zs, acts, starts = [], [], []
    rng = np.random.default_rng(0)
    for ep in range(n_eps):
        env = crafter.Env(seed=7000 + ep); o = env.reset()
        for t in range(max_steps):
            chw = o.transpose(2, 0, 1)
            z = enc(torch.from_numpy(chw[None]).to(dev).float() / 255.0)  # (1,D)
            a = int(rng.integers(17))
            zs.append(z.squeeze(0)); acts.append(a); starts.append(1 if t == 0 else 0)
            o, r, done, _ = env.step(a)
            if done: break
    return torch.stack(zs), np.array(acts, np.int64), np.array(starts, np.int64)


def sample_ctx(z, acts, starts, B, C, rng):
    """Random C-length windows not crossing an episode start."""
    n = len(acts); out = []
    while len(out) < B:
        s = int(rng.integers(0, n - C - 1))
        if starts[s + 1:s + C].sum() == 0:  # no episode boundary inside the window
            out.append(s)
    idx = np.array(out)[:, None] + np.arange(C)[None, :]
    return z[idx], torch.from_numpy(acts[idx]).to(dev)


def run(ec, z, acts, starts, model, D, dh, updates=6000, B=256, C=16, H=16):
    agent = ActorCriticAgent(shim_config(D, dh, entropy_coef=ec), 17, dev).to(dev)
    rng = np.random.default_rng(0); traj = []
    for st in range(1, updates + 1):
        z_ctx, a_ctx = sample_ctx(z, acts, starts, B, C, rng)
        lat, act, logit, rew, term, gf, mus, sgf = imagine(model, agent, z_ctx, a_ctx, H, dev)
        agent.update(lat, act, logit, None, None, None, rew, term, logger=None, global_step=st)
        if st % 500 == 0 or st == 1:
            p = torch.softmax(logit.float(), -1); ent = (-(p * (p + 1e-9).log()).sum(-1)).mean().item()
            traj.append((st, ent))
    return traj


def main():
    ck = torch.load(os.path.join(REPO, "ckpt", "jepa_wm_online_ls_v1.pt"), map_location=dev, weights_only=False)
    D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    dh = model.d_model
    z, acts, starts = collect_frames(enc)
    print(f"contexts: {len(acts)} frames | max entropy ln(17)={np.log(17):.3f}", flush=True)
    print(f"{'entropy_coef':>12} | policy entropy vs #updates (collapse = drops toward 0)", flush=True)
    for ec in [3e-3, 1e-2, 3e-2]:
        traj = run(ec, z, acts, starts, model, D, dh)
        line = "  ".join(f"{st}:{e:.2f}" for st, e in traj)
        print(f"{ec:>12.0e} | {line}", flush=True)


if __name__ == "__main__":
    main()
