"""Track B synthetic (2026-10-15 evening, CPU): persistent state + inferred transformations vs per-step regeneration.

World: a ring of N = 32 cells seen through a camera; classes 0-2 terrain, 3 mob, 4 placed. Per step an action a in
{stay, left, right, do}: left / right scroll the view by one cell (all content shifts one slot, deterministic); do turns the
cell in front of the player (slot 16) into 'placed' (a deterministic edit). Then each mob swaps with a neighbouring non-mob
cell with probability 0.25 per side (chance). Tokens = class embedding (dim 16) + a global brightness vector that jumps to a
new random value with probability 0.25 per step (unpredictable nuisance, as Craftax's quantized daylight) + noise 0.05.
The chance outcome is SUPPLIED to both models (true reassignment applied to the afterstate), isolating persistence.
  REGEN    afterstate_j = sum_c softmax(f)_jc cand_c over {x_j, x_{j-1}, x_{j+1}, gen_j}; gen_j = MLP(local window, action);
           point loss (MSE) through the true reassignment                                     (CorrT principle)
  PERSIST  afterstate = exact copy shifted by a hard global shift choice (straight-through over {-1, 0, +1}, from a head on
           the action), except at cells where a hard edit gate fires (gen_j instead). Gate trained toward the Bayesian
           posterior q_j = sigmoid(gain_j / s + logit p_gate_j) with gain_j = |copy_j - z_j| - |gen_j - z_j| (an edit must
           pay its description cost; s = the frame's mean residual); point loss through the gated afterstate.
Evaluation: H16 recursion from true context (own outputs fed back; true chance outcomes applied each step). Decoding: nearest
class embedding after removing the frame's mean token (invariant to the global nuisance). Accuracy by depth at mob, terrain
and placed cells; the share of mob cells still decoded as mobs (persistence).
Mode "partial": during TRAINING only 70% of the true mob moves are supplied (the rest stay as unexplained change in the
target), mimicking a posterior that misses moves; recursion as above with the full true reassignment. Tests whether soft
regeneration fades when the deterministic target keeps residual uncertainty, and whether MDL-gated persistence does not.
Usage: persist_toy.py [partial] -> artifacts/eda/whiteboard/persist_toy[_partial].json
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[3] / 'artifacts/eda/whiteboard'
N, D, B, FRONT = 32, 16, 64, 16


def step_world(cls, action, g):
    """cls [B,N] long, action [B] -> next cls, true reassignment perm [B,N] (afterstate index -> next), afterstate cls"""
    shift = torch.where(action == 1, -1, torch.where(action == 2, 1, 0))
    after = torch.stack([cls[b].roll(int(shift[b])) for b in range(len(cls))])
    after = torch.where((action == 3)[:, None] & (torch.arange(N)[None] == FRONT), torch.full_like(after, 4), after)
    perm = torch.arange(N).repeat(len(cls), 1)
    nxt = after.clone()
    for b in range(len(cls)):
        used = set()
        for j in torch.where(after[b] == 3)[0].tolist():
            u = float(torch.rand(1, generator=g))
            k = (j + 1) % N if u < .25 else ((j - 1) % N if u < .5 else None)
            if k is None or j in used or k in used or int(after[b, k]) == 3:
                continue
            used |= {j, k}; nxt[b, j], nxt[b, k] = after[b, k], after[b, j]; perm[b, j], perm[b, k] = k, j
    return nxt, perm, shift


def render(cls, bright, emb, g):
    return emb[cls] + bright[:, None] + .05 * torch.randn(*cls.shape, D, generator=g)


class Regen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = torch.nn.Sequential(torch.nn.Linear(3 * D + 4 + 1, 64), torch.nn.GELU())
        self.f = torch.nn.Linear(64, 4); self.gen = torch.nn.Linear(64, D)

    def forward(self, x, action):
        nb = torch.stack([x, x.roll(1, 1), x.roll(-1, 1)], -2)
        pos = (torch.arange(N) == FRONT).float()[None, :, None].expand(len(x), -1, -1)       # same position input as PERSIST
        h = self.ctx(torch.cat([nb.flatten(-2), F.one_hot(action, 4).float()[:, None].expand(-1, N, -1), pos], -1))
        cand = torch.cat([nb, self.gen(h)[..., None, :]], -2)
        return (self.f(h).softmax(-1)[..., None] * cand).sum(-2)


class Persist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.nn.Linear(4, 3)
        self.ctx = torch.nn.Sequential(torch.nn.Linear(3 * D + 4 + 1, 64), torch.nn.GELU())
        self.gen = torch.nn.Linear(64, D); self.gate = torch.nn.Linear(64, 1)
        with torch.no_grad():
            self.gate.bias.fill_(-4.0)

    def forward(self, x, action, z=None):
        oh = F.one_hot(action, 4).float()
        p = self.shift(oh).softmax(-1)                                     # global shift over (-1, 0, +1)
        hard = F.one_hot(p.argmax(-1), 3).float(); st = hard + p - p.detach()
        shifted = torch.stack([x.roll(-1, 1), x, x.roll(1, 1)], 1)         # content index j <- j+1, j, j-1
        copy = (st[:, :, None, None] * shifted).sum(1)
        pos = (torch.arange(N) == FRONT).float()[None, :, None].expand(len(x), -1, -1)
        nb = torch.stack([copy, copy.roll(1, 1), copy.roll(-1, 1)], -2)
        h = self.ctx(torch.cat([nb.flatten(-2), oh[:, None].expand(-1, N, -1), pos], -1))
        gen = self.gen(h); glogit = self.gate(h)[..., 0]
        if z is None:                                                       # recursion: the gate's own decision
            g = (glogit > 0).float()
            return torch.where(g[..., None] > 0, gen, copy), None
        with torch.no_grad():
            res = (copy - z).norm(dim=-1)
            gain = res - (gen - z).norm(dim=-1)
            q = torch.sigmoid(gain / res.mean(1, keepdim=True).clamp_min(1e-6) + glogit)
            g = (q > .5).float()
        out = torch.where(g[..., None] > 0, gen, copy)
        return out, F.binary_cross_entropy_with_logits(glogit, q) + 0 * glogit.sum()


def decode(x, emb):
    xc = x - x.mean(1, keepdim=True)
    ec = emb - emb.mean(0, keepdim=True)
    return (xc[..., None, :] - ec).norm(dim=-1).argmin(-1)


RECALL = 1.0


def run(kind, seed):
    g = torch.Generator().manual_seed(seed); torch.manual_seed(seed)
    emb = torch.randn(5, D, generator=g)
    m = Regen() if kind == 'REGEN' else Persist()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    def batch():
        cls = torch.randint(0, 3, (B, N), generator=g)
        for b in range(B):
            cls[b, torch.randperm(N, generator=g)[:4]] = 3
        action = torch.randint(0, 4, (B,), generator=g)
        bright = torch.randn(B, D, generator=g) * .3
        nb = torch.where(torch.rand(B, 1, generator=g) < .25, torch.randn(B, D, generator=g) * .3, bright)
        nxt, perm, _ = step_world(cls, action, g)
        if RECALL < 1:                                                      # hide some true moves from the supplied chance
            keep = perm.clone()
            for b in range(B):
                for j in range(N):
                    k = int(perm[b, j])
                    if k > j and float(torch.rand(1, generator=g)) > RECALL:
                        keep[b, j], keep[b, k] = j, k
            perm = keep
        return render(cls, bright, emb, g), action, render(nxt, nb, emb, g), perm
    for u in range(3000):
        x, a, z, perm = batch()
        if kind == 'REGEN':
            aft = m(x, a); extra = 0
        else:
            aft, extra = m(x, a, z.gather(1, perm.argsort(1)[..., None].expand(-1, -1, D)))   # posterior vs the pre-chance target
        out = aft.gather(1, perm[..., None].expand(-1, -1, D))
        loss = (out - z).square().mean() + extra
        opt.zero_grad(); loss.backward(); opt.step()
    # recursion
    with torch.no_grad():
        cls = torch.randint(0, 3, (256, N), generator=g)
        for b in range(256):
            cls[b, torch.randperm(N, generator=g)[:4]] = 3
        bright = torch.randn(256, D, generator=g) * .3
        x = render(cls, bright, emb, g)
        acc = []
        for k in range(16):
            a = torch.randint(0, 4, (256,), generator=g)
            nxt, perm, _ = step_world(cls, a, g)
            aft = m(x, a) if kind == 'REGEN' else m(x, a)[0]
            x = aft.gather(1, perm[..., None].expand(-1, -1, D))
            cls = nxt
            d = decode(x, emb)
            row = {'depth': k + 1}
            for name, c in (('mob', 3), ('terrain', None), ('placed', 4)):
                mask = (cls == c) if c is not None else (cls < 3)
                row[name] = round(float((d[mask] == cls[mask]).float().mean()), 3) if mask.any() else None
            row['mob_count_ratio'] = round(float((d == 3).sum() / (cls == 3).sum()), 3)
            xc = x - x.mean(1, keepdim=True); ec = emb - emb.mean(0, keepdim=True)
            row['mob_strength'] = round(float(F.cosine_similarity(xc[cls == 3], ec[3][None]).mean()), 4)    # attenuation analogue
            row['placed_strength'] = round(float(F.cosine_similarity(xc[cls == 4], ec[4][None]).mean()), 4) if (cls == 4).any() else None
            acc.append(row)
    return acc


def main():
    import sys
    global RECALL
    partial = sys.argv[1:] == ['partial']
    RECALL = .7 if partial else 1.0
    res = {}
    for kind in ('REGEN', 'PERSIST'):
        for seed in (0, 1):
            r = run(kind, seed)
            res[f'{kind}_s{seed}'] = r
            print(json.dumps({'model': kind, 'seed': seed, 'd1': r[0], 'd4': r[3], 'd8': r[7], 'd16': r[15]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'persist_toy{"_partial" if partial else ""}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
