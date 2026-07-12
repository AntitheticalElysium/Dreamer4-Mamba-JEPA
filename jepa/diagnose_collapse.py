"""Diagnose the high-ratio collapse WITHOUT another long run. Probes the COLLAPSED checkpoint
(online_hr_v1) against the WORKING one (online_ls_v1, +1.67) as a control, to separate the
candidate causes:
  (1) model exploitation -> imagined 16-step return >> real 16-step return (gap), for collapsed only.
  (2) would the u_s GATE even fire? -> mean head-disagreement u_s during the rollout. If it's not
      elevated on the collapsed (exploiting) rollout, the gate is blind (the Delta-flatness failure mode).
  (3) policy entropy collapse -> low action entropy at eval.
  (4) WM degradation -> worse next-embedding L1 ratio.
All cheap (no training). Compare COLLAPSED vs WORKING columns.
"""
import os, sys, numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "third_party", "Drama"))
import crafter
from conv_encoder import ConvEncoder
from dynamics import JEPADynamics
from agents import ActorCriticAgent
from train_agent import shim_config
dev = "cuda"


def load(wm, ag):
    ck = torch.load(os.path.join(REPO, "ckpt", wm), map_location=dev, weights_only=False); D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev).eval(); enc.load_state_dict(ck["encoder"])
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev).eval(); model.load_state_dict(ck["model"])
    agent = ActorCriticAgent(shim_config(D, model.d_model), 17, dev).to(dev).eval()
    agent.load_state_dict(torch.load(os.path.join(REPO, "ckpt", ag), map_location=dev, weights_only=False))
    return enc, model, agent, D


@torch.no_grad()
def real_rollout(enc, model, agent, seeds, max_steps=400):
    """Run policy in real env; return per-episode (frames_chw, actions, rewards) + mean action entropy + ach."""
    trajs, ents, achs = [], [], []
    for sd in seeds:
        env = crafter.Env(seed=int(sd)); o = env.reset()
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        fs, acts, rews, ac = [], [], [], set()
        for _ in range(max_steps):
            chw = o.transpose(2, 0, 1)
            z = enc(torch.from_numpy(chw[None]).to(dev).float() / 255.0)
            action, logits = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
            p = F.softmax(logits.float(), -1); ents.append(-(p * (p + 1e-9).log()).sum(-1).item())
            _, h, states = model.step(z.unsqueeze(1), action, states)
            o, r, done, info = env.step(int(action.item()))
            fs.append(chw); acts.append(int(action.item())); rews.append(r)
            for k, v in info.get("achievements", {}).items():
                if v > 0: ac.add(k)
            if done: break
        trajs.append((np.array(fs), np.array(acts), np.array(rews, dtype=np.float32))); achs.append(len(ac))
    return trajs, float(np.mean(ents)), float(np.mean(achs))


@torch.no_grad()
def imagined_vs_real(enc, model, agent, trajs, C=16, H=16, nwin=80):
    """For random windows: prime WM on real ctx, roll the POLICY H steps in imagination, decode
    imagined reward + u_s; compare to the REAL reward over the same next-H steps."""
    rng = np.random.default_rng(0); imag, realr, usm = [], [], []
    for _ in range(nwin):
        fs, acts, rews = trajs[rng.integers(len(trajs))]
        if len(fs) < C + H + 1: continue
        s = int(rng.integers(0, len(fs) - C - H))
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        for t in range(C):
            z = enc(torch.from_numpy(fs[s + t:s + t + 1]).to(dev).float() / 255.0)
            preds, h, states = model.step(z.unsqueeze(1), torch.tensor([acts[s + t]], device=dev), states)
        z_cur = preds.mean(0)[:, 0]                         # first imagined latent
        imr, us = 0.0, []
        for i in range(H):
            action, _ = agent.sample(torch.cat([z_cur, h.squeeze(1)], -1), greedy=False)
            preds, h, states = model.step(z_cur.unsqueeze(1), action, states)
            rl, _ = model.reward_term(h); imr += model.symlog.decode(rl).item()
            us.append(preds.float().var(0).mean().item())
            z_cur = preds.mean(0)[:, 0]
        imag.append(imr); realr.append(float(rews[s + C:s + C + H].sum())); usm.append(float(np.mean(us)))
    return float(np.mean(imag)), float(np.mean(realr)), float(np.mean(usm))


@torch.no_grad()
def wm_ratio(enc, model, trajs, T=32, nb=64):
    rng = np.random.default_rng(1); ratios = []
    long_trajs = [t for t in trajs if len(t[0]) > T + 1]
    for _ in range(nb):
        fs, acts, _ = long_trajs[rng.integers(len(long_trajs))]
        s = int(rng.integers(0, len(fs) - T))
        z = enc(torch.from_numpy(fs[s:s + T]).to(dev).float() / 255.0).unsqueeze(0)
        a = torch.tensor(acts[s:s + T], device=dev).unsqueeze(0)
        preds, _ = model(z, a); pred = preds.float().mean(0)[:, :-1]; tgt = z[:, 1:]
        ratios.append((F.l1_loss(pred, tgt) / F.l1_loss(z[:, :-1], tgt)).item())
    return float(np.mean(ratios))


def main():
    seeds = range(5000, 5006)
    for tag, wm, ag in [("COLLAPSED hr_v1", "jepa_wm_online_hr_v1.pt", "jepa_agent_online_hr_v1.pt"),
                        ("WORKING  ls_v1", "jepa_wm_online_ls_v1.pt", "jepa_agent_online_ls_v1.pt")]:
        enc, model, agent, D = load(wm, ag)
        trajs, ent, ach = real_rollout(enc, model, agent, seeds)
        imr, realr, us = imagined_vs_real(enc, model, agent, trajs)
        ratio = wm_ratio(enc, model, trajs)
        maxent = np.log(17)
        print(f"\n=== {tag} ===")
        print(f"  real achievements   : {ach:.2f}")
        print(f"  policy action entropy: {ent:.3f}  (max {maxent:.3f}; low => entropy collapse)")
        print(f"  imagined 16-step ret : {imr:+.3f}")
        print(f"  REAL     16-step ret : {realr:+.3f}")
        print(f"  EXPLOIT GAP (imag-real): {imr-realr:+.3f}  (big +ve => WM over-predicts => exploitation)")
        print(f"  mean u_s on rollout  : {us:.5f}  (gate fires on HIGH u_s; compare across rows)")
        print(f"  WM L1 ratio          : {ratio:.3f}  (<1 good; ~equal across rows => WM not degraded)")
        del enc, model, agent; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
