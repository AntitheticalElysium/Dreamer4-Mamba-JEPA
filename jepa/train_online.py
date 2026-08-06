"""Online Dreamer loop in the JEPA latent — the engine. Alternates:
  (1) COLLECT: act in real Crafter with the current policy (filtering through the WM),
      encode frames with frozen V-JEPA (motion), append to a growing replay buffer;
  (2) TRAIN WM: dynamics + reward + termination on replay sequences (WM is NOT frozen
      here — it keeps learning where the policy actually goes, which closes the model-
      exploitation gap that the offline/frozen setup suffered);
  (3) IMAGINE-TRAIN the policy on replay contexts (reuse train_agent.imagine + AC.update);
  (4) periodically EVAL the greedy policy in real Crafter vs the running random baseline.

Seeds the WM + replay from the offline (random) run so it starts from ratio~0.51, not
scratch. Human-dataset BC-pretraining can seed the policy later (accelerator).
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

import crafter
from agents import ActorCriticAgent
from dynamics import JEPADynamics, curvature_loss
from vjepa_encoder import VJEPAEncoder
from train_agent import shim_config, imagine


class Replay:
    """Flat, episode-aware replay of standardized latents + action/reward/term."""
    def __init__(self, dim, cap=200000, device="cuda"):
        self.z = torch.zeros(cap, dim, device=device)
        self.a = torch.zeros(cap, dtype=torch.long, device=device)
        self.r = torch.zeros(cap, device=device)
        self.d = torch.zeros(cap, device=device)
        self.n, self.cap, self.device = 0, cap, device

    def add(self, z, a, r, d):
        i = self.n % self.cap
        self.z[i] = z; self.a[i] = a; self.r[i] = r; self.d[i] = d
        self.n += 1

    def seed(self, z, a, r, d):
        m = len(a)
        self.z[:m] = z; self.a[:m] = a[:m]; self.r[:m] = r[:m]; self.d[:m] = d[:m]
        self.n = m

    def sample_seq(self, bs, T):
        hi = min(self.n, self.cap) - T - 1
        s = torch.randint(0, hi, (bs,), device=self.device)
        idx = s[:, None] + torch.arange(T, device=self.device)[None, :]
        return self.z[idx], self.a[idx], self.r[idx], self.d[idx]

    def sample_ctx(self, bs, C):
        hi = min(self.n, self.cap) - C - 1
        s = torch.randint(0, hi, (bs,), device=self.device)
        idx = s[:, None] + torch.arange(C, device=self.device)[None, :]
        return self.z[idx], self.a[idx]


@torch.inference_mode()
def collect(env_state, model, agent, enc, replay, mean, std, n_steps, dev, greedy=False):
    env, obs, prev, states, h, prev_a = env_state
    reward_sum = 0.0
    for _ in range(n_steps):
        pf = obs if prev is None else prev
        z_raw = enc.encode(np.stack([pf, obs]), motion=True)[1:2].to(dev)   # (1,D)
        z = ((z_raw - mean) / std)                                          # standardized
        agent_in = torch.cat([z, h.squeeze(1)], dim=-1)
        action, _ = agent.sample(agent_in, greedy=greedy)
        a = int(action.item())
        nobs, r, done, _ = env.step(a)
        replay.add(z.squeeze(0), action.squeeze(0), torch.tensor(float(r), device=dev),
                   torch.tensor(float(done), device=dev))
        _, h, states = model.step(z.unsqueeze(1), action, states)          # advance WM state
        reward_sum += r
        prev, obs = obs, nobs
        if done:
            obs = env.reset(); prev = None
            states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
    return (env, obs, prev, states, h, a), reward_sum


@torch.inference_mode()
def evaluate(model, agent, enc, mean, std, dev, episodes=8, max_steps=400):
    scores, achs = [], []
    for ep in range(episodes):
        env = crafter.Env(seed=1000 + ep); obs = env.reset(); prev = None
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        tot, ach = 0.0, set()
        for _ in range(max_steps):
            pf = obs if prev is None else prev
            z = (enc.encode(np.stack([pf, obs]), motion=True)[1:2].to(dev) - mean) / std
            action, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)  # stochastic = the real metric
            _, h, states = model.step(z.unsqueeze(1), action, states)
            prev = obs                      # current frame becomes 'previous' for next motion clip
            obs, r, done, info = env.step(int(action.item())); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: ach.add(k)
            if done: break
        scores.append(tot); achs.append(len(ach))
    return float(np.mean(scores)), float(np.mean(achs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_data", type=str, default=os.path.join(REPO, "data", "crafter_motion_full.npz"))
    ap.add_argument("--wm_init", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm.pt"))
    ap.add_argument("--agent_init", type=str, default="")
    ap.add_argument("--gate_pct", type=float, default=-1.0)  # >=0 enables u_s-gating at that quantile
    ap.add_argument("--out_tag", type=str, default="online")
    ap.add_argument("--kl_coef", type=float, default=0.0)
    ap.add_argument("--wm_loss", type=str, default="mse", choices=["mse", "cosine"])  # Dreamer-CDP: neg-cosine next-embed
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--wm_updates", type=int, default=40)
    ap.add_argument("--agent_updates", type=int, default=20)
    ap.add_argument("--seq", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm_init, map_location=dev, weights_only=False)
    D, A = ck["latent_dim"], ck["action_dim"]
    mean = torch.tensor(ck["mean"], device=dev); std = torch.tensor(ck["std"], device=dev)
    model = JEPADynamics(latent_dim=D, action_dim=A, n_heads=5).to(dev)
    model.load_state_dict(ck["model"])
    wm_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    agent = ActorCriticAgent(shim_config(D, model.d_model), A, dev).to(dev)
    if args.agent_init:
        agent.load_state_dict(torch.load(args.agent_init, map_location=dev, weights_only=False))
        print(f"initialized agent from BC: {args.agent_init}")
    prior_actor = None
    if args.kl_coef > 0 and args.agent_init:
        import copy
        prior_actor = copy.deepcopy(agent.actor).eval()
        for p in prior_actor.parameters(): p.requires_grad_(False)
        print(f"KL-to-BC prior enabled, kl_coef={args.kl_coef}")
    enc = VJEPAEncoder(res=256, pool="mean", device=dev)

    replay = Replay(D, device=dev)
    d = np.load(args.seed_data)
    zt = torch.tensor((d["latents"][:-1].astype(np.float32) - ck["mean"]) / ck["std"], device=dev)
    replay.seed(zt, torch.tensor(d["actions"], device=dev),
                torch.tensor(d["rewards"], device=dev), torch.tensor(d["terminals"].astype(np.float32), device=dev))
    print(f"seeded replay with {replay.n} random transitions; WM ratio~0.51 start")

    env = crafter.Env(seed=0); obs = env.reset()
    es = (env, obs, None, model.init_state(1, 4096, dev), torch.zeros(1, 1, model.d_model, device=dev), 0)

    for it in range(1, args.iters + 1):
        es, coll_reward = collect(es, model, agent, enc, replay, mean, std, args.collect, dev, greedy=False)
        # --- train WM (dyn + reward + term) ---
        model.train()
        for _ in range(args.wm_updates):
            z, a, r, dn = replay.sample_seq(32, args.seq)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds, h = model(z, a)
                tgt = z.roll(-1, 1).unsqueeze(0).expand_as(preds)  # target[t] = z[t+1] (stop-grad: replay latents)
                if args.wm_loss == "cosine":
                    pred_loss = (1.0 - F.cosine_similarity(preds, tgt, dim=-1)).mean()  # Dreamer-CDP -cos(SG(u'),û)
                else:
                    pred_loss = F.mse_loss(preds, tgt)
                rl, tl = model.reward_term(h)
                loss = pred_loss + 0.01 * curvature_loss(preds.mean(0)) + model.symlog(rl, r) + \
                    F.binary_cross_entropy_with_logits(tl, dn)
            wm_opt.zero_grad(); loss.backward(); wm_opt.step()
        model.eval()
        # --- imagine-train agent ---
        gp = args.gate_pct if args.gate_pct >= 0 else None
        gfs = []
        for _ in range(args.agent_updates):
            z_ctx, a_ctx = replay.sample_ctx(192, args.ctx)
            lat, act, old_logits, rew, term, gf, *_ = imagine(model, agent, z_ctx, a_ctx, args.horizon, dev, gate_pct=gp)
            agent.update(lat, act, old_logits, None, None, None, rew, term, logger=None, global_step=it, prior_actor=prior_actor, kl_coef=args.kl_coef)
            gfs.append(gf)
        if it % 10 == 0 or it == 1:
            sc, ach = evaluate(model, agent, enc, mean, std, dev)
            onp = max(0, replay.n - 30000)
            print(f"iter {it:3d}  replay {replay.n} (onpolicy {onp})  collectR {coll_reward:+.1f} "
                  f"gate {np.mean(gfs):.2f} | EVAL reward {sc:+.2f} achievements {ach:.2f}", flush=True)

    torch.save({"model": model.state_dict(), "mean": ck["mean"], "std": ck["std"],
                "action_dim": A, "latent_dim": D}, os.path.join(REPO, "ckpt", f"jepa_wm_{args.out_tag}.pt"))
    torch.save(agent.state_dict(), os.path.join(REPO, "ckpt", f"jepa_agent_{args.out_tag}.pt"))
    print("saved online WM + agent")


if __name__ == "__main__":
    main()
