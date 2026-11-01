"""T1 structures (T1_PROTOCOL.md). All share the symbolic token embedding and the output heads; they differ only in the
memory structure between them. Input per frame: 63 map-cell tokens (block + mob + position) and 1 HUD token (vitals,
inventory, sleep, light); the action taken from the frame is an extra token. Output at step t: logits for frame t+1.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from mamba_ssm import Mamba

D = 128
NB, NM, NV = 17, 5, 10                                   # block ids, mob codes, values 0..9 (vitals / capped inventory)


class Block(nn.Module):
    def __init__(self, d=D, heads=4):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, kv=None):
        y = self.n1(x); k = y if kv is None else kv
        x = x + self.att(y, k, k, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.block, self.mob = nn.Embedding(NB, D), nn.Embedding(NM, D)
        self.pos = nn.Parameter(torch.randn(65, D) * 0.02)          # 63 cells, HUD, action
        self.vit = nn.ModuleList([nn.Embedding(NV, D) for _ in range(4)])
        self.inv = nn.ModuleList([nn.Embedding(NV, D) for _ in range(12)])
        self.misc = nn.Linear(2, D)
        self.act = nn.Embedding(18, D)                              # 17 actions + "none" (17)

    def forward(self, view, mob, vit, inv, sleep, light, act):
        """view, mob [B, T, 63]; vit [B, T, 4]; inv [B, T, 12]; sleep, light [B, T]; act [B, T] -> [B, T, 65, D]."""
        cells = self.block(view.long()) + self.mob(mob.long())
        hud = sum(e(vit[..., i].long().clamp(0, 9)) for i, e in enumerate(self.vit))
        hud = hud + sum(e(inv[..., i].long().clamp(0, 9)) for i, e in enumerate(self.inv))
        hud = hud + self.misc(torch.stack([sleep.float(), light.float()], -1))
        a = self.act(torch.where(act < 0, torch.full_like(act, 17), act).long())
        return torch.cat([cells, hud[:, :, None], a[:, :, None]], 2) + self.pos


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, NB + NM))
        self.hud = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 16 * NV))

    def forward(self, tok):                                         # tok [B, T, 65, D]
        c = self.cell(tok[:, :, :63])
        h = self.hud(tok[:, :, 63]).view(*tok.shape[:2], 16, NV)
        return c[..., :NB], c[..., NB:], h[..., :4, :], h[..., 4:, :]


def per_token_mamba(m, x):
    B, T, P, d = x.shape
    y = m(x.permute(0, 2, 1, 3).reshape(B * P, T, d))
    return y.view(B, P, T, d).permute(0, 2, 1, 3)


class Global(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.heads = Embed(), Heads()
        self.down = nn.Linear(65 * 16, 2 * D); self.sq = nn.Linear(D, 16)
        self.m1, self.m2 = Mamba(2 * D), Mamba(2 * D)
        self.n1, self.n2 = nn.LayerNorm(2 * D), nn.LayerNorm(2 * D)
        self.up = nn.Linear(2 * D, 65 * D // 4); self.lift = nn.Linear(D // 4, D)
        self.out_pos = nn.Parameter(torch.randn(65, D) * 0.02)

    def forward(self, batch, true_shift=None):
        x = self.emb(*batch)
        B, T = x.shape[:2]
        g = self.down(self.sq(x).reshape(B, T, -1))
        g = g + self.m1(self.n1(g)); g = g + self.m2(self.n2(g))
        tok = self.lift(self.up(g).view(B, T, 65, D // 4)) + self.out_pos
        return self.heads(tok)


class Screen(nn.Module):
    """Per-cell tokens, spatial attention, one Mamba sequence per token position (screen-anchored), spatial attention."""

    def __init__(self):
        super().__init__()
        self.emb, self.heads = Embed(), Heads()
        self.s1, self.s2 = Block(), Block()
        self.m = Mamba(D); self.nm = nn.LayerNorm(D)

    def core(self, x):
        B, T, P, d = x.shape
        x = self.s1(x.reshape(B * T, P, d)).view(B, T, P, d)
        x = x + per_token_mamba(self.m, self.nm(x))
        return x

    def forward(self, batch, true_shift=None):
        x = self.core(self.emb(*batch))
        B, T, P, d = x.shape
        return self.heads(self.s2(x.reshape(B * T, P, d)).view(B, T, P, d))


class Retrieval(nn.Module):
    """Spatial attention, then every token attends to all tokens of the last 16 frames (causal window)."""

    def __init__(self, window=16):
        super().__init__()
        self.emb, self.heads = Embed(), Heads()
        self.s1, self.s2 = Block(), Block()
        self.t1 = Block(); self.W = window
        self.tpos = nn.Parameter(torch.randn(window, D) * 0.02)

    def forward(self, batch, true_shift=None):
        x = self.emb(*batch)
        B, T, P, d = x.shape
        x = self.s1(x.reshape(B * T, P, d)).view(B, T, P, d)
        pad = torch.cat([x.new_zeros(B, self.W - 1, P, d), x], 1)            # [B, T + W - 1, P, d]
        kv = pad.unfold(1, self.W, 1).permute(0, 1, 4, 2, 3)                  # [B, T, W, P, d]
        kv = (kv + self.tpos[None, None, :, None]).reshape(B * T, self.W * P, d)
        x = self.t1(x.reshape(B * T, P, d), kv=self.t1.n1(kv)).view(B, T, P, d)
        return self.heads(self.s2(x.reshape(B * T, P, d)).view(B, T, P, d))


class Slots(Screen):
    """Screen plus K recurrent object slots: one slot-attention step per frame (softmax over slots), GRU update, and the
    tokens read the slots back by cross-attention before the last spatial block."""

    def __init__(self, K=8):
        super().__init__()
        self.K = K
        self.init = nn.Parameter(torch.randn(K, D) * 0.02)
        self.q, self.k, self.v = nn.Linear(D, D), nn.Linear(D, D), nn.Linear(D, D)
        self.gru = nn.GRUCell(D, D); self.sn = nn.LayerNorm(D); self.tn = nn.LayerNorm(D)
        self.read = Block()

    def forward(self, batch, true_shift=None):
        x = self.core(self.emb(*batch))
        B, T, P, d = x.shape
        s = self.init.expand(B, -1, -1).contiguous(); outs = []
        k, v = self.k(self.tn(x)), self.v(self.tn(x))
        for t in range(T):
            att = torch.einsum("bkd,bpd->bkp", self.q(self.sn(s)), k[:, t]) / d ** 0.5
            att = att.softmax(1) + 1e-6                                          # competition over slots
            upd = torch.einsum("bkp,bpd->bkd", att / att.sum(2, keepdim=True), v[:, t])
            s = self.gru(upd.reshape(B * self.K, d), s.reshape(B * self.K, d)).view(B, self.K, d)
            outs.append(s)
        S = torch.stack(outs, 1)                                                 # [B, T, K, d]
        x = self.read(x.reshape(B * T, P, d), kv=self.read.n1(S.reshape(B * T, self.K, d))).view(B, T, P, d)
        return self.heads(self.s2(x.reshape(B * T, P, d)).view(B, T, P, d))


SHIFTS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


class EgoMemory(Screen):
    """Screen plus a 25 x 25 feature canvas anchored by self-motion. Each frame: the canvas is shifted by a soft mixture of
    the 9 one-cell offsets (predicted from the previous and current frame and the previous action; or the true offset for
    E*), the current view is written at the centre through a gate, and the tokens read the whole canvas by
    cross-attention. No position labels are used by E."""

    def __init__(self):
        super().__init__()
        self.wr = nn.Linear(D, 64); self.gate = nn.Linear(D, 1)
        self.shift = nn.Sequential(nn.Linear(4 * D, D), nn.GELU(), nn.Linear(D, 9))
        self.cpos = nn.Parameter(torch.randn(625, 64) * 0.02); self.cup = nn.Linear(64, D)
        self.read = Block()

    @staticmethod
    def roll(M, dr, dc):
        """Content moves by (-dr, -dc) when the player moves by (dr, dc); zero fill."""
        out = torch.zeros_like(M)
        H, W = M.shape[1:3]
        rs, re = max(0, -dr), H - max(0, dr); cs, ce = max(0, -dc), W - max(0, dc)
        out[:, rs:re, cs:ce] = M[:, rs + dr:re + dr, cs + dc:ce + dc]
        return out

    def forward(self, batch, true_shift=None):
        x = self.core(self.emb(*batch))
        B, T, P, d = x.shape
        M = x.new_zeros(B, 25, 25, 64); prev = x[:, 0]; reads = []; probs = []
        for t in range(T):
            cur = x[:, t]
            feat = torch.cat([cur[:, :63].mean(1), prev[:, :63].mean(1), cur[:, 64], (cur[:, :63] - prev[:, :63]).abs().mean(1)], -1)
            if true_shift is not None:
                pi = F.one_hot(true_shift[:, t], 9).float()
            else:
                pi = self.shift(feat).softmax(-1) if t > 0 else F.one_hot(torch.full((B,), 4, device=x.device), 9).float()
            probs.append(pi)
            M = sum(pi[:, i, None, None, None] * self.roll(M, dr, dc) for i, (dr, dc) in enumerate(SHIFTS))
            w = self.wr(cur[:, :63]).view(B, 7, 9, 64); g = torch.sigmoid(self.gate(cur[:, :63])).view(B, 7, 9, 1)
            M = M.clone(); M[:, 9:16, 8:17] = g * w + (1 - g) * M[:, 9:16, 8:17]
            reads.append(self.cup(M.reshape(B, 625, 64) + self.cpos)); prev = cur
        R = torch.stack(reads, 1)                                                # [B, T, 625, D]
        x = self.read(x.reshape(B * T, P, d), kv=self.read.n1(R.reshape(B * T, 625, d))).view(B, T, P, d)
        self.last_shift_probs = torch.stack(probs, 1)
        return self.heads(self.s2(x.reshape(B * T, P, d)).view(B, T, P, d))


def make(name):
    return {"G": Global, "S": Screen, "A": Retrieval, "O": Slots, "E": EgoMemory, "Estar": EgoMemory}[name]()
