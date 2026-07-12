"""Imagination training in the JEPA latent: train an actor-critic policy PURELY inside
the frozen Mamba-JEPA world model (the "Dreamer" step). Reuses DRAMA's ActorCriticAgent
+ calc_lambda_return VERBATIM via a shim config (feat_dim = z_jepa 768 + h_mamba 512).

The imagination rollout mirrors DRAMA's imagine_data2 buffer layout exactly (to avoid
off-by-one divergence): sample_buffer/dist_feat_buffer length H+1, action/reward/term
length H; reward+termination decoded from dist_feat_buffer[:, :-1].

Metric: imagined return (mean predicted reward per rollout) should rise as the policy
learns to exploit the world model — the signature of learning in imagination.
"""
import os
import sys
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from box import Box
from agents import ActorCriticAgent
from dynamics import JEPADynamics


def shim_config(latent_dim, hidden_dim, entropy_coef=3e-4):
    """Minimal config so DRAMA's ActorCriticAgent gets feat_dim = latent_dim+hidden_dim
    and DreamerV3 hyperparameters (matched to DRAMA's configure.yaml). entropy_coef is
    tunable: DRAMA's 3e-4 is weak; a high policy-update ratio can collapse entropy without more."""
    return Box({
        "BasicSettings": {"Use_amp": True},
        "Models": {
            "WorldModel": {"CategoricalDim": latent_dim, "ClassDim": 1, "HiddenStateDim": hidden_dim},
            "Agent": {
                "Unimix_ratio": 0.0,
                "AC": {"NumLayers": 3, "Gamma": 0.985, "Lambda": 0.95, "EntropyCoef": entropy_coef,
                       "Max_grad_norm": 100, "Warmup_steps": 1000, "Act": "SiLU",
                       "Optimiser": "Laprop", "Laprop": {"LearningRate": 4e-5, "Epsilon": 1e-20},
                       "Actor": {"HiddenUnits": 256}, "Critic": {"HiddenUnits": 512}},
            },
        },
    })


def _roll_latent(preds, sample_head=True):
    """preds (K,B,1,D) -> next latent (B,1,D). sample_head=True picks a random RPF head
    per batch element = STOCHASTIC imagination (diverse futures; heads agree in-dist,
    diverge OOD, so it's noisy exactly where the WM is uncertain). False = deterministic mean."""
    if not sample_head:
        return preds.mean(0)
    K, B = preds.shape[0], preds.shape[1]
    idx = torch.randint(0, K, (B,), device=preds.device)
    return preds[idx, torch.arange(B, device=preds.device)]


@torch.no_grad()
def imagine(model, agent, z_ctx, a_ctx, H, dev, sample_head=True, gate_pct=None, shadow_pct=0.9, noise=0.0):
    """Roll the policy inside the frozen WM. z_ctx (B,C,D), a_ctx (B,C).
    Returns tensors for agent.update: latent[z,h] (B,H+1,D+Dh), action (B,H),
    old_logits (B,H,A), reward (B,H), termination (B,H). Mirrors imagine_data2.

    gate_pct (AHEAD-style u_s gating): if set, compute per-step head-disagreement u_s
    and TRUNCATE the rollout (termination=1) at/after the first step whose u_s exceeds
    the gate_pct-quantile of this batch's u_s -> the policy never trains on imagination
    beyond where the WM becomes unreliable (anti-exploitation). Also returns gate_frac."""
    B, C, D = z_ctx.shape
    Dh = model.d_model
    states = model.init_state(B, C + H + 2, dev)
    model.set_delta_capture(False)
    # prime context with real (z,a)
    for t in range(C):
        preds, h, states = model.step(z_ctx[:, t:t + 1], a_ctx[:, t], states)
    z_cur = _roll_latent(preds, sample_head)                                   # (B,1,D): predicted first imagined z
    sample_buf = torch.zeros(B, H + 1, D, device=dev)
    feat_buf = torch.zeros(B, H + 1, Dh, device=dev)
    act_buf = torch.zeros(B, H, dtype=torch.long, device=dev)
    logit_buf = torch.zeros(B, H, agent.action_dim, device=dev)
    sample_buf[:, 0] = z_cur.squeeze(1)
    feat_buf[:, 0] = h.squeeze(1)
    us_list = []
    for i in range(H):
        agent_in = torch.cat([sample_buf[:, i], feat_buf[:, i]], dim=-1)  # (B, D+Dh)
        action, logits = agent.sample(agent_in)                          # (B,), (B,A)
        act_buf[:, i] = action
        logit_buf[:, i] = logits
        preds, h, states = model.step(z_cur, action, states)   # action (B,), like context priming
        us_list.append(preds.float().var(0).mean(-1).squeeze(1))         # (B,) head-disagreement u_s
        z_cur = _roll_latent(preds, sample_head)
        if noise > 0:  # inject stochasticity into imagined states (proxy for a stochastic WM) -> anti-overfit
            z_cur = z_cur + noise * z_cur.std(dim=-1, keepdim=True) * torch.randn_like(z_cur)
        sample_buf[:, i + 1] = z_cur.squeeze(1)
        feat_buf[:, i + 1] = h.squeeze(1)
    # reward + termination from dist_feat_buffer[:, :-1]  (H predictions)
    rew_logits, term_logit = model.reward_term(feat_buf[:, :-1])
    reward = model.symlog.decode(rew_logits)                             # (B,H)
    termination = (term_logit > 0).float()                              # (B,H)
    us = torch.stack(us_list, dim=1)                                      # (B,H) head-disagreement
    mean_us = us.mean().item()
    # SHADOW-GATE: what fraction of imagined steps WOULD be truncated at shadow_pct -- computed
    # ALWAYS, never applied. Lets us eyeball whether the u_s gate would fire on a bad run.
    sthr = torch.quantile(us.reshape(-1), shadow_pct)
    shadow_gate_frac = torch.cummax((us > sthr).float(), dim=1).values.mean().item()
    gate_frac = 0.0
    if gate_pct is not None:                                             # ACTUAL gate (only if enabled)
        thr = torch.quantile(us.reshape(-1), gate_pct)
        gate = torch.cummax((us > thr).float(), dim=1).values            # 1 at/after first spike
        termination = torch.clamp(termination + gate, max=1.0)
        gate_frac = gate.mean().item()
    latent = torch.cat([sample_buf, feat_buf], dim=-1)                   # (B,H+1,D+Dh)
    return latent, act_buf, logit_buf, reward, termination, gate_frac, mean_us, shadow_gate_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=os.path.join(REPO, "data", "crafter_vjepa.npz"))
    ap.add_argument("--wm", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm.pt"))
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--imagine_bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm, map_location=dev, weights_only=False)
    model = JEPADynamics(latent_dim=ck["latent_dim"], action_dim=ck["action_dim"], n_heads=5).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                          # frozen WM; train policy in its dream

    d = np.load(args.data)
    lat = (d["latents"].astype(np.float32) - ck["mean"]) / ck["std"]
    act = d["actions"].astype(np.int64)
    L = torch.tensor(lat, device=dev); A = torch.tensor(act, device=dev)
    n = len(act)

    agent = ActorCriticAgent(shim_config(ck["latent_dim"], model.d_model), ck["action_dim"], dev)
    agent = agent.to(dev)  # DRAMA moves actor/critic individually; ensure symlog bins buffer is on device too
    print(f"agent feat_dim={ck['latent_dim']+model.d_model} action_dim={ck['action_dim']}  "
          f"params={sum(p.numel() for p in agent.parameters())/1e6:.2f}M")

    rng = np.random.default_rng(0)
    ret_hist = []
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, n - args.ctx - 1, size=args.imagine_bs)
        z_ctx = torch.stack([L[s:s + args.ctx] for s in starts])
        a_ctx = torch.stack([A[s:s + args.ctx] for s in starts])
        latent, action, old_logits, reward, termination, *_ = imagine(model, agent, z_ctx, a_ctx, args.horizon, dev)
        agent.update(latent, action, old_logits, None, None, None, reward, termination, logger=None, global_step=step)
        ret_hist.append(reward.sum(1).mean().item())         # imagined return per rollout
        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  imagined_return {np.mean(ret_hist[-200:]):+.3f}")

    os.makedirs(os.path.join(REPO, "ckpt"), exist_ok=True)
    torch.save(agent.state_dict(), os.path.join(REPO, "ckpt", "jepa_agent.pt"))
    print("saved agent -> ckpt/jepa_agent.pt")


if __name__ == "__main__":
    main()
