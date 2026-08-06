"""Offline pretraining of the TRAINED-encoder Mamba-JEPA world model on competent-PPO
frames. This is the ground-up rebuild that fixes the Q2 ceiling (frozen V-JEPA exposed
only 0.27 held-out action-info vs a trained CNN's 0.42).

Architecture (every choice sourced, per the no-guessing rule):
  - Context encoder  : ConvEncoder  -- DreamerV3 CNN spec (arXiv 2301.04104): {32,64,128,256},
                       k4/s2, channel-wise LayerNorm + SiLU. (jepa/conv_encoder.py)
  - Target encoder   : EMA(context), stop-grad -- I-JEPA / BYOL (arXiv 2301.08243). tau=0.996.
  - Predictor        : JEPADynamics (Mamba-2) -- unchanged; predicts next EMBEDDING.
  - Losses           : L1 teacher-forcing + L1 2-step rollout -- V-JEPA 2-AC (arXiv 2506.09985,
                       "L_tf + L_rollout", both L1, T=2, fights compounding error);
                       + reward (symlog-two-hot) + termination (BCE) -- DRAMA/DreamerV3 heads;
                       + BC action CE -- grounds the encoder to be control-relevant (anti-collapse
                       backstop #2; #1 is the EMA target). VICReg variance term on standby.
  - Recon-free       : never decode pixels (JEPA), unlike Dreamer 3/4.

Because the encoder is now TRAINED, precomputed embeddings would go stale -> replay/data must
hold RAW FRAMES and encode on the fly (DreamerV3 does this; cheap now that the encoder is 2.8M).

VALIDATION GATE before wiring into the online loop: learned encoder held-out action-decodability
must beat frozen V-JEPA's 0.27 (target ~0.42), WM ratio must be sane, and embedding std must stay
> 0 (no collapse).
"""
import os, sys, glob, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
DRAMA = os.path.join(REPO, "third_party", "Drama")
for p in (HERE, DRAMA):
    if p not in sys.path:
        sys.path.insert(0, p)
from conv_encoder import ConvEncoder, EMATarget, variance_reg
from dynamics import JEPADynamics


def load_ppo_frames(nfiles):
    """-> obs uint8 (S,L,3,64,64), act (S,L), rew (S,L). Verified format: 1000 traj/file,
    each obs (4,3,64,64) float[0,1], actions (4,1), rewards (4,1)."""
    files = sorted(glob.glob(os.path.join(REPO, "data", "crafter_ppo", "ex", "*.pt")))[:nfiles]
    obs, act, rew = [], [], []
    for f in files:
        for t in torch.load(f, map_location="cpu", weights_only=False):
            obs.append((t["obs"].numpy() * 255).clip(0, 255).astype(np.uint8))  # (L,3,64,64)
            act.append(t["actions"].reshape(-1).numpy().astype(np.int64))         # (L,)
            rew.append(t["rewards"].reshape(-1).numpy().astype(np.float32))       # (L,)
    return np.stack(obs), np.stack(act), np.stack(rew)


def encode_seq(enc, obs_u8, dev):
    """obs_u8 (B,L,3,64,64) uint8 -> (B,L,D) float embeddings (grad through enc)."""
    B, L = obs_u8.shape[:2]
    x = obs_u8.reshape(B * L, 3, 64, 64).to(dev).float() / 255.0
    e = enc(x)
    return e.reshape(B, L, -1)


@torch.no_grad()
def probe_action_decode(enc, obs_u8, act, dev, epochs=25, bs=256):
    """Gate metric comparable to Q2's 0.42: freeze enc, train an MLP on e->action, held-out acc."""
    enc.eval()
    # obs_u8 (M,L,3,64,64), act (M,L) -> flatten to frames/actions
    frames = obs_u8.reshape(-1, 3, 64, 64)
    a = torch.from_numpy(act.reshape(-1)).to(dev)
    N = frames.shape[0]
    embs = []
    for i in range(0, N, 1024):
        x = torch.from_numpy(frames[i:i + 1024]).to(dev).float() / 255.0
        embs.append(enc(x).cpu())
    z = torch.cat(embs).to(dev)
    s = int(N * 0.8)
    zm, zs = z[:s].mean(0), z[:s].std(0) + 1e-6; z = (z - zm) / zs
    with torch.enable_grad():
        net = nn.Sequential(nn.Linear(z.shape[1], 512), nn.GELU(), nn.Linear(512, 17)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            idx = rng.permutation(s)
            for i in range(0, s, bs):
                b = idx[i:i + bs]
                loss = F.cross_entropy(net(z[b]), a[b]); opt.zero_grad(); loss.backward(); opt.step()
        te = (net(z[s:]).argmax(-1) == a[s:]).float().mean().item()
    enc.train()
    return te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfiles", type=int, default=40)
    ap.add_argument("--embed_dim", type=int, default=512)   # matches Mamba d_model / DreamerV3-S deter
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)       # DreamerV3 WM lr
    ap.add_argument("--tau", type=float, default=0.996)     # I-JEPA/BYOL EMA
    ap.add_argument("--w_roll", type=float, default=1.0)    # V-JEPA 2-AC: equal TF/rollout
    ap.add_argument("--w_bc", type=float, default=1.0)
    ap.add_argument("--w_var", type=float, default=0.0)     # VICReg backstop, off unless collapse
    ap.add_argument("--out_tag", type=str, default="jepacnn")
    ap.add_argument("--eval_every", type=int, default=2000)
    # Mamba predictor knobs (for the new-baseline ablation; JEPADynamics passthrough)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_state", type=int, default=64)   # SSM state size -- the untested Mamba knob
    ap.add_argument("--ssl_init", type=str, default="")  # load SSL-pretrained ConvBackbone
    ap.add_argument("--freeze_encoder", action="store_true")  # V-JEPA-2-AC style: frozen encoder, train dynamics only
    args = ap.parse_args()
    dev = "cuda"

    obs, act, rew = load_ppo_frames(args.nfiles)
    S, L = obs.shape[:2]
    ntr = int(S * 0.9)
    print(f"segments={S} seglen={L} | train={ntr} val={S-ntr} | embed_dim={args.embed_dim}", flush=True)
    # tensors kept on CPU (uint8), moved per-batch
    obs_t = torch.from_numpy(obs); act_t = torch.from_numpy(act); rew_t = torch.from_numpy(rew)

    enc = ConvEncoder(embed_dim=args.embed_dim).to(dev)
    if args.ssl_init:
        enc.backbone.load_state_dict(torch.load(args.ssl_init, map_location=dev, weights_only=False)["backbone"])
        print(f"loaded SSL-pretrained backbone from {os.path.basename(args.ssl_init)}", flush=True)
    if args.freeze_encoder:
        for p in enc.backbone.parameters(): p.requires_grad_(False)  # freeze SSL backbone; proj stays trainable
        print("encoder BACKBONE frozen (V-JEPA-2-AC style); only proj + dynamics train", flush=True)
    ema = EMATarget(enc, tau=args.tau); ema.target.to(dev)
    model = JEPADynamics(latent_dim=args.embed_dim, action_dim=17, n_heads=5,
                         d_model=args.d_model, n_layers=args.n_layers, d_state=args.d_state).to(dev)
    bc_head = nn.Sequential(nn.Linear(model.d_model, 256), nn.SiLU(), nn.Linear(256, 17)).to(dev)
    params = [p for p in enc.parameters() if p.requires_grad] + list(model.parameters()) + list(bc_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    n_enc = sum(p.numel() for p in enc.parameters()) / 1e6
    n_dyn = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"params: encoder {n_enc:.2f}M + dynamics {n_dyn:.2f}M + bc {sum(p.numel() for p in bc_head.parameters())/1e6:.2f}M", flush=True)

    rng = np.random.default_rng(0)
    t0 = time.time()
    for st in range(1, args.steps + 1):
        enc.train(); model.train()
        idx = rng.integers(0, ntr, size=args.bs)
        ob = obs_t[idx].to(dev); a = act_t[idx].to(dev); r = rew_t[idx].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = encode_seq(enc, ob, dev)                        # (B,L,D) grad
            with torch.no_grad():
                e_tgt = encode_seq(ema.target, ob, dev)         # (B,L,D) stop-grad target
            preds, h = model(e, a)                              # preds (K,B,L,D)
            # (1) teacher-forcing L1: pred e_{t+1} from e_t  [V-JEPA 2-AC]
            tgt = e_tgt[:, 1:].unsqueeze(0)                     # (1,B,L-1,D)
            l_tf = F.l1_loss(preds[:, :, :-1], tgt.expand_as(preds[:, :, :-1]))
            # (2) 2-step rollout L1: feed predicted e_2 back, predict e_3
            e1 = e[:, :1]
            predsA, _ = model(e1, a[:, :1]); ehat2 = predsA.mean(0)[:, 0]        # (B,D) grad
            seq2 = torch.stack([e[:, 0], ehat2], dim=1)
            predsB, _ = model(seq2, a[:, :2]); ehat3 = predsB.mean(0)[:, 1]      # (B,D)
            l_roll = F.l1_loss(ehat3, e_tgt[:, 2])
            # (3) reward (symlog two-hot) + (4) termination (BCE, ~0 mid-episode) -- DRAMA heads
            rl, tl = model.reward_term(h)
            l_rew = model.symlog(rl, r)
            l_term = F.binary_cross_entropy_with_logits(tl, torch.zeros_like(tl))
            # (5) BC action CE -- grounds encoder (anti-collapse #2)
            l_bc = F.cross_entropy(bc_head(h).reshape(-1, 17), a.reshape(-1))
            # (6) VICReg variance backstop (off by default)
            l_var = variance_reg(e.reshape(-1, e.shape[-1])) if args.w_var > 0 else e.new_zeros(())
            loss = l_tf + args.w_roll * l_roll + l_rew + l_term + args.w_bc * l_bc + args.w_var * l_var
        opt.zero_grad(); loss.backward(); opt.step()
        ema.update(enc)

        if st % args.eval_every == 0 or st == 1:
            # WM ratio on val (L1 pred vs persistence baseline) + embedding std (collapse check)
            enc.eval(); model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                vidx = np.arange(ntr, min(S, ntr + 512))
                ob = obs_t[vidx].to(dev); a = act_t[vidx].to(dev)
                e = encode_seq(enc, ob, dev)
                preds, _ = model(e, a)
                pred = preds.float().mean(0)[:, :-1]; tv = e[:, 1:]
                l1_pred = F.l1_loss(pred, tv).item()
                l1_pers = F.l1_loss(e[:, :-1], tv).item()
                estd = e.reshape(-1, e.shape[-1]).std(0).mean().item()
            acc = probe_action_decode(enc, obs[ntr:], act[ntr:], dev)
            dt = time.time() - t0
            print(f"step {st:6d} [{dt:5.0f}s] loss {loss.item():.3f} "
                  f"(tf {l_tf.item():.3f} roll {l_roll.item():.3f} rew {l_rew.item():.3f} bc {l_bc.item():.3f}) "
                  f"| WM ratio {l1_pred/l1_pers:.3f} | embstd {estd:.3f} | HELDOUT act-acc {acc:.3f}", flush=True)

    torch.save({"encoder": enc.state_dict(), "model": model.state_dict(), "bc_head": bc_head.state_dict(),
                "embed_dim": args.embed_dim, "action_dim": 17},
               os.path.join(REPO, "ckpt", f"jepa_wm_{args.out_tag}.pt"))
    print(f"saved -> ckpt/jepa_wm_{args.out_tag}.pt", flush=True)


if __name__ == "__main__":
    main()
