"""BC-pretrain the DRAMA actor on competent-PPO play, in the TRAINED-encoder latent.
Mirrors train_bc.py exactly (incl. the prefix-hidden fix that avoids a_t label leakage and
matches the online eval semantics [z_t, h_{t-1}]), but produces z by encoding the raw PPO
frames with our trained ConvEncoder instead of reading frozen V-JEPA latents.

Encoder + Mamba are FROZEN here (BC trains the actor only); the online loop unfreezes the WM.
Old BC on the 0.27-info frozen latent reached ~3.2 achievements; on this 0.44-info latent it
should do materially better -- a direct test of the rebuild's payoff.
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)

from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from agents import ActorCriticAgent
from train_agent import shim_config
from train_jepa_wm import load_ppo_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm_jepacnn_v1.pt"))
    ap.add_argument("--nfiles", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default=os.path.join(REPO, "ckpt", "jepa_agent_bc_jepa.pt"))
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    for p in list(enc.parameters()) + list(model.parameters()):
        p.requires_grad_(False)

    obs, act, _ = load_ppo_frames(args.nfiles)          # obs (S,L,3,64,64) uint8, act (S,L)
    S, L = obs.shape[:2]
    # encode all frames once with the frozen trained encoder
    with torch.no_grad():
        zs = []
        for i in range(0, S, 256):
            ob = torch.from_numpy(obs[i:i + 256]).to(dev).float().reshape(-1, 3, 64, 64) / 255.0
            zs.append(enc(ob).reshape(min(256, S - i), L, D).cpu())
        z = torch.cat(zs).to(dev)
    a = torch.from_numpy(act).to(dev)
    print(f"BC data: {S} segments x {L} steps ; latent D={D}", flush=True)

    agent = ActorCriticAgent(shim_config(D, model.d_model), 17, dev).to(dev)
    opt = torch.optim.AdamW(agent.actor.parameters(), lr=args.lr)     # BC trains the ACTOR only
    rng = np.random.default_rng(0)
    for ep in range(1, args.epochs + 1):
        idx = rng.permutation(S); losses, accs = [], []
        for i in range(0, S, args.bs):
            b = idx[i:i + args.bs]
            zb, ab = z[b], a[b]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.backbone(zb, ab)                            # h[t] consumed z_t,a_t (leaks a_t!)
                hp = torch.zeros_like(h); hp[:, 1:] = h[:, :-1]        # PREFIX h[t-1] = state BEFORE a_t
                logits = agent.actor(torch.cat([zb, hp], -1)).float()  # matches eval: [z_t, h_prev]
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ab.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); accs.append((logits.argmax(-1) == ab).float().mean().item())
        print(f"epoch {ep}  BC loss {np.mean(losses):.3f}  action-acc {np.mean(accs):.3f}", flush=True)

    torch.save(agent.state_dict(), args.out)
    print(f"saved BC agent -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
