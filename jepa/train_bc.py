"""BC-pretrain the policy on competent PPO play (the accelerator / our 'good gameplay'
seed, Dreamer 4 style). For each 4-step segment: run the frozen WM to get the Mamba
hidden h, form the agent input [z, h], and train the actor to imitate the PPO action
(cross-entropy). Seeds a competent policy so the online loop starts above random.

Keeps Mamba + JEPA fixed (the experiment). Latents standardized with the WM's own
mean/std so the policy lives in the same space it will act in.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents import ActorCriticAgent
from dynamics import JEPADynamics
from train_agent import shim_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join(REPO, "data", "crafter_ppo_seg.npz"))
    ap.add_argument("--wm", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm.pt"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "ckpt", "jepa_agent_bc.pt"))
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    D = ck["latent_dim"]
    model = JEPADynamics(latent_dim=D, action_dim=ck["action_dim"], n_heads=5).to(dev).eval()
    model.load_state_dict(ck["model"])
    for p in model.parameters(): p.requires_grad_(False)
    mean = torch.tensor(ck["mean"], device=dev); std = torch.tensor(ck["std"], device=dev)

    d = np.load(args.data)
    z = (torch.tensor(d["seg_latents"], device=dev) - mean) / std   # (S,T,D)
    a = torch.tensor(d["seg_actions"], device=dev)                  # (S,T)
    S, T = a.shape
    print(f"BC data: {S} segments x {T} steps ; action_dim {ck['action_dim']}")

    agent = ActorCriticAgent(shim_config(D, model.d_model), ck["action_dim"], dev).to(dev)
    opt = torch.optim.AdamW(agent.actor.parameters(), lr=args.lr)   # BC trains the ACTOR only
    rng = np.random.default_rng(0)

    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(S); losses, accs = [], []
        for i in range(0, S, args.bs):
            b = idx[i:i + args.bs]
            zb, ab = z[b], a[b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.backbone(zb, ab)                         # h[t] consumed z_t,a_t (leaks a_t!)
                hp = torch.zeros_like(h); hp[:, 1:] = h[:, :-1]     # PREFIX hidden h[t-1] = state BEFORE a_t
                logits = agent.actor(torch.cat([zb, hp], -1)).float()  # matches eval: [z_t, h_prev]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ab.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            accs.append((logits.argmax(-1) == ab).float().mean().item())
        print(f"epoch {ep}  BC loss {np.mean(losses):.3f}  action-acc {np.mean(accs):.3f}")

    torch.save(agent.state_dict(), args.out)
    print(f"saved BC agent -> {args.out}")


if __name__ == "__main__":
    main()
