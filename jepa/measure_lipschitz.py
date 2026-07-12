"""Measure the Lipschitz constants from 'On Training in Imagination' (arXiv 2605.06732) on our
GOOD (ls_v1,+1.67) vs DEGRADED (full_v1,+0.02) checkpoints. The theorem: imagination return-error
blows up as gamma*L_f*(1+L_pi) -> 1. If the degraded model has higher L_f (and/or the contraction
factor >= 1), the failure is the uncontrolled-dynamics-Lipschitz one the paper characterizes, and the
fix (spectral norm + temporal-straightening curvature, which we currently DON'T use) is grounded.

L_f (one-step dynamics Lipschitz): perturb the current embedding e by delta, measure
||next-embed(e+delta) - next-embed(e)|| / ||delta|| via the parallel backbone (no recurrent-state issues).
L_pi (policy Lipschitz): perturb the actor input [z,h], measure ||logits change|| / ||delta||.
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
dev = "cuda"; GAMMA = 0.985


def load(wm, ag):
    ck = torch.load(os.path.join(REPO, "ckpt", wm), map_location=dev, weights_only=False); D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    agent = ActorCriticAgent(shim_config(D, model.d_model), 17, dev).to(dev).eval()
    agent.load_state_dict(torch.load(os.path.join(REPO, "ckpt", ag), map_location=dev, weights_only=False))
    return enc, model, agent, D


@torch.no_grad()
def collect_seqs(enc, model, agent, n=6, T=16, maxs=300):
    """Real short embedding sequences (B,T,D) + actions (B,T) + the [z,h] actor inputs seen."""
    E, A, ZH = [], [], []
    for sd in range(5000, 5000 + n):
        env = crafter.Env(seed=sd); o = env.reset(); states = model.init_state(1, 4096, dev)
        h = torch.zeros(1, 1, model.d_model, device=dev); es, as_, zhs = [], [], []
        for _ in range(maxs):
            z, _ = enc_one(enc, o, dev)
            zhs.append(torch.cat([z, h.squeeze(1)], -1).squeeze(0))
            act, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
            _, h, states = model.step(z.unsqueeze(1), act, states)
            es.append(z.squeeze(0)); as_.append(int(act.item()))
            o, r, done, _ = env.step(int(act.item()))
            if done: break
        es = torch.stack(es); as_ = torch.tensor(as_, device=dev)
        for s in range(0, len(as_) - T, T):
            E.append(es[s:s + T]); A.append(as_[s:s + T])
        ZH += zhs
    return torch.stack(E), torch.stack(A), torch.stack(ZH)


@torch.no_grad()
def lipschitz_f(model, E, A, eps=1e-2, ndir=5):
    """One-step dynamics Lipschitz: perturb last embedding, measure next-embed prediction change."""
    preds0, _ = model(E, A); base = preds0.float().mean(0)[:, -1]        # next-embed pred at last step
    ratios = []
    for _ in range(ndir):
        d = torch.randn_like(E[:, -1]); d = eps * d / (d.norm(dim=-1, keepdim=True) + 1e-9)
        Ep = E.clone(); Ep[:, -1] = Ep[:, -1] + d
        predsp, _ = model(Ep, A); pert = predsp.float().mean(0)[:, -1]
        ratios.append(((pert - base).norm(dim=-1) / eps))
    r = torch.stack(ratios)
    return r.mean().item(), r.max().item()


@torch.no_grad()
def lipschitz_pi(agent, ZH, eps=1e-2, ndir=5):
    """Policy Lipschitz: perturb actor input [z,h], measure logit change (softmax prob L2)."""
    _, base = agent.sample(ZH, greedy=True); base = F.softmax(base.float(), -1)
    ratios = []
    for _ in range(ndir):
        d = torch.randn_like(ZH); d = eps * d / (d.norm(dim=-1, keepdim=True) + 1e-9)
        _, lp = agent.sample(ZH + d, greedy=True); pert = F.softmax(lp.float(), -1)
        ratios.append(((pert - base).norm(dim=-1) / eps))
    r = torch.stack(ratios)
    return r.mean().item(), r.max().item()


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="*", default=[]); args = ap.parse_args()
    pairs = [(f"{t}", f"jepa_wm_{t}.pt", f"jepa_agent_{t}.pt") for t in args.tags] or [
        ("GOOD ls_v1(+1.67)", "jepa_wm_online_ls_v1.pt", "jepa_agent_online_ls_v1.pt"),
        ("DEGRADED full_v1(+0.02)", "jepa_wm_online_full_v1.pt", "jepa_agent_online_full_v1.pt")]
    for tag, wm, ag in pairs:
        enc, model, agent, D = load(wm, ag)
        E, A, ZH = collect_seqs(enc, model, agent)
        lf_mean, lf_max = lipschitz_f(model, E, A)
        lpi_mean, lpi_max = lipschitz_pi(agent, ZH[:2000])
        contract_mean = GAMMA * lf_mean * (1 + lpi_mean)
        contract_max = GAMMA * lf_max * (1 + lpi_max)
        print(f"\n=== {tag} ===")
        print(f"  L_f  (dynamics)  mean {lf_mean:.3f}  max {lf_max:.3f}")
        print(f"  L_pi (policy)    mean {lpi_mean:.3f}  max {lpi_max:.3f}")
        print(f"  gamma*L_f*(1+L_pi)  mean {contract_mean:.3f}  max {contract_max:.3f}   (>=1 => error bound BLOWS UP)")
        del enc, model, agent; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
