"""Online Dreamer loop with the TRAINED-encoder Mamba-JEPA world model (the correct arch,
built ground-up after Q2 showed frozen V-JEPA is the ceiling: 0.27 vs 0.44 action-info).

Differs from train_online.py (frozen V-JEPA) in three sourced ways:
  - encoder is a TRAINED ConvEncoder (+ EMA target), kept learning online -> replay stores RAW
    FRAMES (uint8) and encodes on the fly (precomputed embeddings would go stale). [DreamerV3]
  - WM training = L1 teacher-forcing + 2-step rollout + reward/term, with EMA target update.
    [V-JEPA 2-AC (arXiv 2506.09985) + I-JEPA EMA]
  - policy (DRAMA ActorCriticAgent) imagines in the learned embedding space via imagine() [reused].
Warm-started from the offline checkpoint (ckpt/jepa_wm_jepacnn_v1.pt). No external latent
standardization: the agent's VecNormalize handles it, and the WM was trained on raw embeddings.
GroupNorm/RMSNorm everywhere (no BatchNorm) so train/eval mode is a no-op during imagination.
"""
import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)

import crafter
from conv_encoder import ConvEncoder, EMATarget
from dynamics import JEPADynamics, curvature_loss
from agents import ActorCriticAgent
from train_agent import shim_config, imagine


class FrameReplay:
    """Episode-aware replay of RAW FRAMES (uint8 CHW) + action/reward/done, on CPU."""
    def __init__(self, cap, dev):
        self.obs = np.zeros((cap, 3, 64, 64), np.uint8)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)
        self.n, self.cap, self.dev = 0, cap, dev

    def add(self, o_chw, a, r, d):
        i = self.n % self.cap
        self.obs[i] = o_chw; self.a[i] = a; self.r[i] = r; self.d[i] = d; self.n += 1

    def _starts(self, T, k, rng):
        hi = min(self.n, self.cap) - T - 1
        out = []
        for s in rng.integers(0, hi, size=k * 3):
            if self.d[s:s + T - 1].sum() == 0:   # reject windows that cross an episode boundary
                out.append(int(s))
            if len(out) >= k:
                break
        while len(out) < k:
            out.append(int(rng.integers(0, hi)))
        return np.array(out[:k])

    def sample_seq(self, bs, T, rng):
        idx = self._starts(T, bs, rng)[:, None] + np.arange(T)[None, :]
        return (torch.from_numpy(self.obs[idx]).to(self.dev),
                torch.from_numpy(self.a[idx]).to(self.dev),
                torch.from_numpy(self.r[idx]).to(self.dev),
                torch.from_numpy(self.d[idx]).to(self.dev))

    def sample_ctx(self, bs, C, rng):
        o, a, _, _ = self.sample_seq(bs, C, rng)
        return o, a


def enc_seq(enc, obs_u8):
    """(B,T,3,64,64) uint8 -> (B,T,D) embeddings."""
    B, T = obs_u8.shape[:2]
    return enc(obs_u8.reshape(B * T, 3, 64, 64).float() / 255.0).reshape(B, T, -1)


def enc_one(enc, obs_hwc, dev):
    """crafter obs (64,64,3) uint8 -> (1,D) embedding, and (3,64,64) uint8 for replay."""
    chw = obs_hwc.transpose(2, 0, 1)
    z = enc(torch.from_numpy(chw[None]).to(dev).float() / 255.0)
    return z, chw


@torch.no_grad()
def collect(es, enc, model, agent, env, replay, n_steps, dev):
    obs, states, h = es
    rsum = 0.0
    for _ in range(n_steps):
        z, chw = enc_one(enc, obs, dev)
        action, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
        a = int(action.item())
        nobs, r, done, _ = env.step(a)
        replay.add(chw, a, float(r), float(done))
        _, h, states = model.step(z.unsqueeze(1), action, states)
        rsum += r; obs = nobs
        if done:
            obs = env.reset(); states = model.init_state(1, 4096, dev)
            h = torch.zeros(1, 1, model.d_model, device=dev)
    return (obs, states, h), rsum


@torch.no_grad()
def evaluate(enc, model, agent, dev, episodes=8, max_steps=400):
    was_training = agent.training
    agent.eval()   # CRITICAL: freeze VecNormalize stats -> honest deployment eval (matches eval_jepa).
    #              In train mode VecNormalize adapts to eval states on the fly and HID a degenerate policy.
    scores, achs = [], []
    for ep in range(episodes):
        env = crafter.Env(seed=1000 + ep); obs = env.reset()
        states = model.init_state(1, 4096, dev); h = torch.zeros(1, 1, model.d_model, device=dev)
        tot, ach = 0.0, set()
        for _ in range(max_steps):
            z, _ = enc_one(enc, obs, dev)
            action, _ = agent.sample(torch.cat([z, h.squeeze(1)], -1), greedy=False)
            _, h, states = model.step(z.unsqueeze(1), action, states)
            obs, r, done, info = env.step(int(action.item())); tot += r
            for k, v in info.get("achievements", {}).items():
                if v > 0: ach.add(k)
            if done: break
        scores.append(tot); achs.append(len(ach))
    if was_training:
        agent.train()
    return float(np.mean(scores)), float(np.mean(achs))


def wm_update(enc, ema, model, bc_head, opt, replay, seq_T, bs, rng, dev, w_roll=1.0, curv_w=0.0):
    o, a, r, d = replay.sample_seq(bs, seq_T, rng)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        e = enc_seq(enc, o)
        with torch.no_grad():
            e_tgt = enc_seq(ema.target, o)
        preds, h = model(e, a)
        l_tf = F.l1_loss(preds[:, :, :-1], e_tgt[:, 1:].unsqueeze(0).expand_as(preds[:, :, :-1]))
        # 2-step rollout (V-JEPA 2-AC): predict e_3 from e_1 via predicted e_2
        predsA, _ = model(e[:, :1], a[:, :1]); ehat2 = predsA.mean(0)[:, 0]
        seq2 = torch.stack([e[:, 0], ehat2], dim=1)
        predsB, _ = model(seq2, a[:, :2]); ehat3 = predsB.mean(0)[:, 1]
        l_roll = F.l1_loss(ehat3, e_tgt[:, 2])
        rl, tl = model.reward_term(h)
        l_rew = model.symlog(rl, r)
        l_term = F.binary_cross_entropy_with_logits(tl, d)
        l_bc = F.cross_entropy(bc_head(h).reshape(-1, 17), a.reshape(-1))
        # temporal-straightening (Wang 2026 / On-Training-in-Imagination Prop.1): lowers the dynamics
        # velocity Lipschitz L_f -> tightens the compounding-error bound gamma*L_f*(1+L_pi).
        l_curv = curvature_loss(preds.mean(0)) if curv_w > 0 else e.new_zeros(())
        loss = l_tf + w_roll * l_roll + l_rew + l_term + l_bc + curv_w * l_curv
    opt.zero_grad(); loss.backward(); opt.step(); ema.update(enc)
    return loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm_init", type=str, default=os.path.join(REPO, "ckpt", "jepa_wm_jepacnn_v1.pt"))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--collect", type=int, default=500)
    ap.add_argument("--wm_updates", type=int, default=40)
    ap.add_argument("--agent_updates", type=int, default=20)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--imagine_bs", type=int, default=512)  # DRAMA ImagineBatchSize 1024; raise policy replay ratio
    ap.add_argument("--gate_pct", type=float, default=-1.0)
    ap.add_argument("--entropy_coef", type=float, default=3e-4)  # DRAMA default; bounds policy Lipschitz L_pi
    ap.add_argument("--target_ent", type=float, default=0.0)     # >0: auto-tune entropy_coef to hold polENT here (bounds L_pi)
    ap.add_argument("--curv_w", type=float, default=0.0)         # temporal-straightening weight -> bounds L_f
    ap.add_argument("--imag_noise", type=float, default=0.0)     # inject stochasticity into imagined states (anti-overfit proxy)
    ap.add_argument("--spectral", action="store_true")          # spectral-norm the Mamba/heads -> bounds L_f
    ap.add_argument("--shadow_pct", type=float, default=0.9)     # shadow-gate percentile (logs, never fires)
    ap.add_argument("--lr_wm", type=float, default=1e-4)
    ap.add_argument("--freeze_encoder", action="store_true")  # keep SSL-pretrained backbone frozen (general); train Mamba
    ap.add_argument("--agent_init", type=str, default="")  # warm-start actor from BC (move+do priors)
    ap.add_argument("--out_tag", type=str, default="online_jepa")
    ap.add_argument("--save_every", type=int, default=0)   # periodic checkpoint every N iters (0=off)
    args = ap.parse_args()
    dev = "cuda"

    ck = torch.load(args.wm_init, map_location=dev, weights_only=False)
    D = ck["embed_dim"]
    enc = ConvEncoder(embed_dim=D).to(dev); enc.load_state_dict(ck["encoder"])
    if args.freeze_encoder:
        for p in enc.backbone.parameters(): p.requires_grad_(False)   # keep general SSL backbone frozen
        print("SSL encoder BACKBONE frozen online; Mamba learns dynamics on long sequences", flush=True)
    ema = EMATarget(enc, tau=0.996); ema.target.to(dev)
    model = JEPADynamics(latent_dim=D, action_dim=17, n_heads=5).to(dev); model.load_state_dict(ck["model"])
    bc_head = nn.Sequential(nn.Linear(model.d_model, 256), nn.SiLU(), nn.Linear(256, 17)).to(dev)
    bc_head.load_state_dict(ck["bc_head"])
    wm_params = [p for p in enc.parameters() if p.requires_grad] + list(model.parameters()) + list(bc_head.parameters())
    wm_opt = torch.optim.AdamW(wm_params, lr=args.lr_wm)
    agent = ActorCriticAgent(shim_config(D, model.d_model, entropy_coef=args.entropy_coef), 17, dev).to(dev)
    if args.agent_init:
        agent.load_state_dict(torch.load(args.agent_init, map_location=dev, weights_only=False))
        print(f"actor warm-started from {os.path.basename(args.agent_init)}", flush=True)
    print(f"warm-started from {os.path.basename(args.wm_init)} | feat_dim={D+model.d_model} "
          f"| WM+enc params {sum(p.numel() for p in wm_params)/1e6:.2f}M "
          f"| agent {sum(p.numel() for p in agent.parameters())/1e6:.2f}M", flush=True)

    replay = FrameReplay(200000, dev)
    env = crafter.Env(seed=0); obs = env.reset()
    es = (obs, model.init_state(1, 4096, dev), torch.zeros(1, 1, model.d_model, device=dev))
    rng = np.random.default_rng(0)
    gp = args.gate_pct if args.gate_pct >= 0 else None
    t0 = time.time()

    for it in range(1, args.iters + 1):
        es, coll_r = collect(es, enc, model, agent, env, replay, args.collect, dev)
        enc.train(); model.train()
        for _ in range(args.wm_updates):
            wm_update(enc, ema, model, bc_head, wm_opt, replay, args.seq, 32, rng, dev, curv_w=args.curv_w)
        enc.eval(); model.eval()
        gfs, imret, ents, muss, sgfs = [], [], [], [], []
        for _ in range(args.agent_updates):
            o_ctx, a_ctx = replay.sample_ctx(args.imagine_bs, args.ctx, rng)
            with torch.no_grad():
                z_ctx = enc_seq(enc, o_ctx)
            lat, act, logit, rew, term, gf, mus, sgf = imagine(model, agent, z_ctx, a_ctx, args.horizon, dev,
                                                               gate_pct=gp, shadow_pct=args.shadow_pct, noise=args.imag_noise)
            agent.update(lat, act, logit, None, None, None, rew, term, logger=None, global_step=it)
            gfs.append(gf); imret.append(rew.sum(1).mean().item())
            p = torch.softmax(logit.float(), -1); ents.append((-(p * (p + 1e-9).log()).sum(-1)).mean().item())
            muss.append(mus); sgfs.append(sgf)                          # shadow-gate signals (u_s, would-be gate frac)
        # AUTOMATIC ENTROPY TARGETING (SAC-style): hold polENT at target_ent -> robustly bounds the policy
        # Lipschitz L_pi so gamma*L_f*(1+L_pi) stays < 1 regardless of ratio/length (On-Training-in-Imagination).
        if args.target_ent > 0:
            cur_ent = float(np.mean(ents))
            agent.entropy_coef = float(np.clip(agent.entropy_coef * np.exp(0.5 * (args.target_ent - cur_ent)), 1e-5, 0.2))
        if it % 10 == 0 or it == 1:
            sc, ach = evaluate(enc, model, agent, dev)
            print(f"iter {it:3d} [{time.time()-t0:5.0f}s] replay {replay.n} collectR {coll_r:+.1f} "
                  f"| polENT {np.mean(ents):.3f} entC {agent.entropy_coef:.1e} imR {np.mean(imret):+.2f} u_s {np.mean(muss):.4f} "
                  f"shadowGate {np.mean(sgfs):.2f} gate {np.mean(gfs):.2f} | EVAL reward {sc:+.2f} ach {ach:.2f}", flush=True)
        if args.save_every > 0 and it % args.save_every == 0:   # periodic checkpoint (resume/eval-the-peak, survive kills)
            torch.save({"encoder": enc.state_dict(), "model": model.state_dict(), "bc_head": bc_head.state_dict(),
                        "embed_dim": D, "action_dim": 17}, os.path.join(REPO, "ckpt", f"jepa_wm_{args.out_tag}_it{it}.pt"))
            torch.save(agent.state_dict(), os.path.join(REPO, "ckpt", f"jepa_agent_{args.out_tag}_it{it}.pt"))
            print(f"  [ckpt @ it{it}]", flush=True)

    torch.save({"encoder": enc.state_dict(), "model": model.state_dict(), "bc_head": bc_head.state_dict(),
                "embed_dim": D, "action_dim": 17}, os.path.join(REPO, "ckpt", f"jepa_wm_{args.out_tag}.pt"))
    torch.save(agent.state_dict(), os.path.join(REPO, "ckpt", f"jepa_agent_{args.out_tag}.pt"))
    print(f"saved -> ckpt/jepa_wm_{args.out_tag}.pt + agent", flush=True)


if __name__ == "__main__":
    main()
