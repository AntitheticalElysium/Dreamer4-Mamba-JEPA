"""T2c (T2_PROTOCOL.md, Amendment 1): t2_joint.py plus two counter-mechanisms to entity erasure.
  --enc-lr-scale f   encoder learning rate = f x the predictor's (Tang et al. 2023 "faster predictor"; Dreamer-CDP ~1/67)
  --glimpse w        grounded dense target: at every predicted frame, the pixels of ONE random 3 x 3-cell window of the
                     real frame are decoded from the predictor's tokens at those cells (linear 128 -> 147 per cell),
                     MSE x w. The fovea idea without learned attention. --glimpse-all decodes all 81 cells (full pixel
                     reconstruction; diagnostic reference only).
Everything else is identical to t2_joint.py (same classes, data, schedule).
T2 joint training (T2_PROTOCOL.md). One run = one (anti-collapse arm, unroll H, seed).

Usage: t2_joint.py --data DIR --out DIR --anti {sg,ema,sigreg,vicreg,none} --H {1,4,16} --seed S [--steps 10000]
Window of W = C + H frames (C = 4). The predictor consumes real latents for frames 0..C-1 and its own predictions after,
so frames 1..C are predicted at depth 1 and frame C-1+k at depth k (k <= H). Every predicted frame gets the
self-prediction loss and the reward / termination losses (uniform weights).
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

C = 4
D_TOK, D_LAT, D_H = 128, 64, 128


class Block(nn.Module):
    def __init__(self, d, heads=4, mlp=2):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, mlp * d), nn.GELU(), nn.Linear(mlp * d, d))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.att(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Encoder(nn.Module):
    """63x63x3 frame -> 81 tokens (7x7 patches) -> 2 transformer blocks -> per-token latent (64)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(147, D_TOK)
        self.pos = nn.Parameter(torch.randn(81, D_TOK) * 0.02)
        self.blocks = nn.Sequential(Block(D_TOK), Block(D_TOK))
        self.out = nn.Linear(D_TOK, D_LAT)

    @staticmethod
    def patches(x):                                                  # x uint8 [N, 63, 63, 3]
        x = x.float() / 255.0
        x = x.view(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 81, 147)
        return x

    def forward(self, x):
        return self.out(self.blocks(self.proj(self.patches(x)) + self.pos))


class Predictor(nn.Module):
    """Recurrent per-token state h [B, 81, 128]; one step consumes a latent (real or predicted) and the action."""

    def __init__(self):
        super().__init__()
        self.inp = nn.Linear(D_LAT, D_H)
        self.act = nn.Embedding(17, D_H)
        self.pos = nn.Parameter(torch.randn(81, D_H) * 0.02)
        self.blocks = nn.Sequential(Block(D_H), Block(D_H))
        self.read = nn.Sequential(nn.LayerNorm(D_H), nn.Linear(D_H, D_LAT))
        self.reward = nn.Sequential(nn.LayerNorm(D_H), nn.Linear(D_H, D_H), nn.GELU(), nn.Linear(D_H, 1))
        self.term = nn.Sequential(nn.LayerNorm(D_H), nn.Linear(D_H, D_H), nn.GELU(), nn.Linear(D_H, 1))

    def step(self, h, z, a):
        x = h + self.inp(F.layer_norm(z, (D_LAT,))) + self.pos
        x = torch.cat([self.act(a)[:, None], x], 1)
        x = self.blocks(x)
        g, h = x[:, 0], x[:, 1:]
        pooled = torch.cat([g[:, None], h], 1).mean(1)
        return h, self.read(h), self.reward(pooled).squeeze(-1), self.term(pooled).squeeze(-1)

    def unroll(self, z_real, actions, n_teacher):
        """z_real [B, W, 81, 64] (only the first n_teacher used as inputs); actions [B, W-1].
        Returns predictions for frames 1..W-1: zhat [B, W-1, 81, 64], rhat, chat [B, W-1]."""
        B, W = actions.shape[0], actions.shape[1] + 1
        h = torch.zeros(B, 81, D_H, device=actions.device)
        zs, rs, cs, hs = [], [], [], []
        z_in = z_real[:, 0]
        for t in range(W - 1):
            h, zhat, r, c = self.step(h, z_in, actions[:, t])
            zs.append(zhat); rs.append(r); cs.append(c); hs.append(h)
            z_in = z_real[:, t + 1] if t + 1 < n_teacher else zhat
        self.last_h = torch.stack(hs, 1)                                       # [B, W-1, 81, D_H] for the glimpse decoder
        return torch.stack(zs, 1), torch.stack(rs, 1), torch.stack(cs, 1)


def sigreg(z, gen):
    """LeJEPA SIGReg (MINIMAL.md estimator), batched over positions. z [P, N, D]: per position, Epps-Pulley statistic of
    256 random unit 1-D projections against N(0, 1), scaled by N; mean over positions and directions."""
    P, N, Dd = z.shape
    t = torch.linspace(0, 3, 17, device=z.device)
    dt = 3 / 16
    w = torch.full((17,), 2 * dt, device=z.device); w[[0, -1]] = dt
    phi = torch.exp(-t.square() / 2)
    A = torch.randn(Dd, 256, device=z.device, generator=gen)
    A = A / A.norm(dim=0)
    xt = (z.float() @ A).unsqueeze(-1) * t                            # [P, N, 256, 17]
    err = (xt.cos().mean(1) - phi).square() + xt.sin().mean(1).square()  # [P, 256, 17]
    return ((err @ (w * phi)) * N).mean()


def vicreg_terms(z):
    """VICReg variance + covariance (main_vicreg.py), batched over positions. z [P, N, D]; relative weights std 1,
    cov 1/25 (VICReg's 25 / 25 / 1 with the prediction term at weight 1)."""
    z = z.float() - z.float().mean(1, keepdim=True)
    std = torch.sqrt(z.var(1) + 1e-4)                                 # [P, D]
    cov = z.transpose(1, 2) @ z / (z.shape[1] - 1)                    # [P, D, D]
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
    return F.relu(1 - std).mean() + (off.pow(2).sum((1, 2)) / z.shape[2]).mean() / 25.0


class Data:
    def __init__(self, d, W, seed):
        self.frames = np.load(os.path.join(d, "train_frames.npy"), mmap_mode="r")
        m = np.load(os.path.join(d, "train_meta.npz"))
        self.actions, self.rewards, self.terms = m["actions"], m["rewards"], m["terms"]
        starts, lens = m["starts"], m["lengths"]
        self.valid = np.concatenate([np.arange(s, s + L - W + 1) for s, L in zip(starts, lens) if L >= W])
        self.W, self.rng = W, np.random.default_rng(seed)

    def batch(self, B, device):
        s = self.rng.choice(self.valid, B)
        idx = s[:, None] + np.arange(self.W)[None]
        fr = torch.from_numpy(np.ascontiguousarray(self.frames[idx.reshape(-1)])).to(device)
        a = torch.from_numpy(self.actions[idx[:, :-1]].astype(np.int64)).to(device)
        r = torch.from_numpy(self.rewards[idx[:, :-1]]).to(device)
        c = torch.from_numpy(self.terms[idx[:, :-1]].astype(np.float32)).to(device)
        return fr, a, r, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--anti", required=True, choices=["sg", "ema", "sigreg", "vicreg", "none"])
    p.add_argument("--H", type=int, required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--batch", type=int, default=32)
    p.add_argument("--save", default="1000,3000,6000,10000")
    p.add_argument("--enc-lr-scale", type=float, default=1.0); p.add_argument("--glimpse", type=float, default=0.0)
    p.add_argument("--glimpse-all", action="store_true")
    p.add_argument("--idm", type=float, default=0.0)          # Amendment 2: encoder inverse dynamics (SLIM, 2610.03137)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = os.environ.get("T2_DEVICE", "cuda")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    enc, pred = Encoder().to(dev), Predictor().to(dev)
    ema = copy.deepcopy(enc).requires_grad_(False) if a.anti == "ema" else None
    torch.save({"enc": enc.state_dict(), "pred": pred.state_dict()}, os.path.join(a.out, "ckpt_00000.pt"))
    dec = nn.Linear(D_H, 147).to(dev)
    idm = nn.Sequential(nn.Linear(3 * D_LAT, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 17)).to(dev)
    heads = list(pred.parameters()) + list(dec.parameters()) + list(idm.parameters())
    params = list(enc.parameters()) + heads
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": 3e-4 * a.enc_lr_scale},
                             {"params": heads, "lr": 3e-4}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500))
    W = C + a.H
    data = Data(a.data, W, a.seed + 1000)
    gen = torch.Generator(device=dev).manual_seed(a.seed + 7)
    saves = {int(x) for x in a.save.split(",")}
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        fr, act, rew, term = data.batch(a.batch, dev)
        B = act.shape[0]
        with torch.autocast(dev, dtype=torch.bfloat16):
            z = enc(fr).view(B, W, 81, D_LAT)
            if a.anti == "ema":
                with torch.no_grad():
                    tgt_raw = ema(fr).view(B, W, 81, D_LAT)
            else:
                tgt_raw = z
            zhat, rhat, chat = pred.unroll(z, act, n_teacher=C)
        tgt = F.layer_norm(tgt_raw[:, 1:].float(), (D_LAT,))
        if a.anti == "sg":
            tgt = tgt.detach()
        l_pred = (zhat.float() - tgt).pow(2).mean()
        l_rew = (rhat.float() - rew).pow(2).mean()
        l_term = F.binary_cross_entropy_with_logits(chat.float(), term)
        loss = l_pred + l_rew + l_term
        l_pix = torch.zeros((), device=dev)
        if a.glimpse > 0:
            pix = Encoder.patches(fr).view(B, W, 81, 147)[:, 1:]                 # real next-frame patches [B, W-1, 81, 147]
            hid = pred.last_h                                                    # [B, W-1, 81, D_H]
            if a.glimpse_all:
                sel = torch.arange(81, device=dev).expand(B, W - 1, 81)
            else:
                r0 = torch.randint(0, 7, (B, W - 1, 1), device=dev); c0 = torch.randint(0, 7, (B, W - 1, 1), device=dev)
                off = torch.tensor([r * 9 + c for r in range(3) for c in range(3)], device=dev)
                sel = (r0 * 9 + c0) + off                                         # [B, W-1, 9]
            g = lambda x: torch.gather(x, 2, sel[..., None].expand(*sel.shape, x.shape[-1]))
            l_pix = (dec(g(hid).float()) - g(pix)).pow(2).mean()
            loss = loss + a.glimpse * l_pix
        l_idm = torch.zeros((), device=dev)
        if a.idm > 0:                                  # predict a_t from encoder latents (z_t, z_t+1); gradient into encoder
            zl = F.layer_norm(z.float(), (D_LAT,))
            f = torch.cat([zl[:, :-1].mean(2), zl[:, 1:].mean(2), (zl[:, 1:] - zl[:, :-1]).abs().mean(2)], -1)
            l_idm = F.cross_entropy(idm(f).reshape(-1, 17), act.reshape(-1))
            loss = loss + a.idm * l_idm
        l_reg = torch.zeros((), device=dev)
        if a.anti in ("sigreg", "vicreg"):
            sub = z[:, :5].reshape(B * 5, 81, D_LAT).float().transpose(0, 1)   # [81 positions, 160 frames, 64]
            l_reg = sigreg(sub, gen) if a.anti == "sigreg" else vicreg_terms(sub)
            loss = loss + (0.09 if a.anti == "sigreg" else 1.0) * l_reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, po in zip(ema.parameters(), enc.parameters()):
                    pe.mul_(0.995).add_(po.detach(), alpha=0.005)
        if step % 100 == 0:
            with torch.no_grad():
                zz = F.layer_norm(z[:, :2].reshape(-1, D_LAT).float(), (D_LAT,))
                std = zz.std(0).mean().item()
            rec = {"step": step, "loss": loss.item(), "pix": l_pix.item(), "idm": l_idm.item(), "pred": l_pred.item(), "rew": l_rew.item(), "term": l_term.item(),
                   "reg": l_reg.item(), "tok_std": std, "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n"); log.flush()
            if step % 1000 == 0:
                print(rec, flush=True)
        if step in saves:
            torch.save({"enc": enc.state_dict(), "pred": pred.state_dict(), "dec": dec.state_dict(), "step": step, "args": vars(a)},
                       os.path.join(a.out, f"ckpt_{step:05d}.pt"))
    json.dump({"done": True, "seconds": time.time() - t0, "args": vars(a)}, open(os.path.join(a.out, "done.json"), "w"))


if __name__ == "__main__":
    main()
