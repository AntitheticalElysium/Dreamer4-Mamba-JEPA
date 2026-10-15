"""Minimal synthetic test of the inferred-chance feedback loop (2026-10-15, CPU, seconds).

World: a 1-D ring of 32 cells. Each cell holds terrain (3 classes, persistent) or a mob (rare, 3/32); tokens = class embedding
(dim 16) + a global per-step drift vector (deterministic, shared by all cells: the 'daylight' analogue) + small noise. Each
step every mob independently swaps with its left or right neighbour with probability 0.25 each (chance); terrain persists.
Model (the transport structure): afterstate a_j = sum over d in {-1,0,+1} softmax(f(x)_j)[d] * x_{j+d} + g(drift context),
i.e. a soft copy over self / neighbours (CorrT analogue, trainable f) plus a learned global drift term. Training uses the
point loss through an inferred reassignment T_hat of a (detached) vs z, as in transport_ic_train.infer:
  gain(j, j+1) = |a_j - z_j| + |a_{j+1} - z_{j+1}| - |a_{j+1} - z_j| - |a_j - z_{j+1}|
  ML   accept greedily when gain > 0                           (the rule used by TRANSPORT_IC)
  MAP  accept when gain > s * log((1 - pi) / pi)               (s = mean residual, pi = running accepted rate)
  ORACLE  the true reassignment (F1b analogue)
Readings over 3,000 updates: inferred swaps per step vs true (0.19 per step expected: 3 mobs x 0.5 x ...), precision against
the true swaps, and the afterstate's identity weight on terrain cells (does it learn to pre-swap persistent content?).
Mode "ghost": the afterstate is first trained 1,500 updates WITHOUT reassignment (identity T: plain regression, which learns
the mean and ghosts mobs onto neighbours, like the D parent), then 3,000 updates of inferred-chance training (MAP / ML /
ORACLE). Reported additionally: the afterstate's ghost mass (weight a mob cell gives to its neighbours' copies, and that
empty neighbours of a mob give to the mob).
Usage: em_toy.py [ghost] -> artifacts/eda/whiteboard/em_toy[_ghost].json
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[3] / 'artifacts/eda/whiteboard'
N, D, B = 32, 16, 64


def world(g, emb):
    """one batch of transitions: x [B,N,D], z [B,N,D], true swap set as left-endpoint mask [B,N]"""
    cls = torch.randint(0, 3, (B, N), generator=g)
    mob = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        mob[b, torch.randperm(N, generator=g)[:3]] = True
    cls = torch.where(mob, torch.full_like(cls, 3), cls)
    drift = torch.randn(B, 1, D, generator=g) * .3
    x = emb[cls] + .05 * torch.randn(B, N, D, generator=g)
    nxt = cls.clone(); swap = torch.zeros(B, N, dtype=torch.bool)
    for b in range(B):
        used = set()
        for j in torch.where(mob[b])[0].tolist():
            u = float(torch.rand(1, generator=g))
            k = (j + 1) % N if u < .25 else ((j - 1) % N if u < .5 else None)
            if k is None or j in used or k in used or bool(mob[b, k]):
                continue
            used |= {j, k}; nxt[b, j], nxt[b, k] = cls[b, k], cls[b, j]
            swap[b, min(j, k) if abs(j - k) == 1 else max(j, k)] = True
    z = emb[nxt] + drift + .05 * torch.randn(B, N, D, generator=g)
    return x, z, drift, swap


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(3 * D, 32), torch.nn.GELU(), torch.nn.Linear(32, 3))
        self.g = torch.nn.Linear(D, D)
        with torch.no_grad():
            self.f[-1].bias.copy_(torch.tensor([0., 3., 0.]))               # start near identity copy

    def forward(self, x, drift):
        nb = torch.stack([x.roll(1, 1), x, x.roll(-1, 1)], -2)              # [B,N,3,D]: left, self, right
        w = self.f(nb.flatten(-2)).softmax(-1)
        return (w[..., None] * nb).sum(-2) + self.g(drift), w


def infer(a, z, rule, pi):
    k_idx = torch.arange(N).roll(-1)
    res = (a - z).norm(dim=-1)
    gain = res + res[:, k_idx] - (a[:, k_idx] - z).norm(dim=-1) - (a - z[:, k_idx]).norm(dim=-1)     # pair (j, j+1)
    s = res.mean(1, keepdim=True)
    bar = torch.zeros_like(gain) if rule == 'ML' else s * math.log((1 - pi) / pi)
    acc = torch.zeros_like(gain, dtype=torch.bool)
    for b in range(len(a)):
        used = set()
        for j in torch.argsort(gain[b], descending=True).tolist():
            if gain[b, j] <= bar[b, 0 if rule != 'ML' else j]:
                break
            k = (j + 1) % N
            if j in used or k in used:
                continue
            used |= {j, k}; acc[b, j] = True
    return acc


def apply(a, acc):
    out = a.clone()
    for b, j in zip(*torch.where(acc)):
        k = (int(j) + 1) % N
        out[b, j], out[b, k] = a[b, k], a[b, j]
    return out


def ghost_mass(m, g, emb):
    with torch.no_grad():
        x, z, drift, swap = world(g, emb)
        a, w = m(x, drift)
        mob = (x - emb[3]).norm(dim=-1) < 1.0                               # cells holding a mob
        left, right = mob.roll(-1, 1), mob.roll(1, 1)                       # cell whose right / left neighbour is a mob
        to_mob = torch.where(left, w[..., 2], torch.zeros_like(w[..., 2])) + torch.where(right, w[..., 0], torch.zeros_like(w[..., 0]))
        nb = (left | right) & ~mob
        return round(float(to_mob[nb].mean()), 4), round(float(w[..., 1][mob].mean()), 4)


def run(rule, seed, ghost=False):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(4, D, generator=g)
    emb[3] *= 1.5                                                         # mobs are distinct
    torch.manual_seed(seed)
    m = Model(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    pi = .02; curve = []
    if ghost:
        for u in range(1500):
            x, z, drift, swap = world(g, emb)
            a, _ = m(x, drift)
            loss = (a - z).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        gm = ghost_mass(m, g, emb)
        curve.append({'update': 0, 'after_regression_ghost_mass_to_mob': gm[0], 'mob_self_weight': gm[1]})
    for u in range(3000):
        x, z, drift, swap = world(g, emb)
        a, w = m(x, drift)
        if rule == 'ORACLE':
            acc = swap
        else:
            with torch.no_grad():
                acc = infer(a.detach(), z, rule, pi)
            pi = min(max(.99 * pi + .01 * float(acc.float().sum(1).mean()) / N, 1e-4), .5)
        loss = (apply(a, acc) - z).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (u + 1) % 250 == 0:
            tp = int((acc & swap).sum()); fp = int((acc & ~swap).sum())
            mob = (x - 0).norm(dim=-1) > 0                                   # placeholder mask (all cells)
            gm = ghost_mass(m, g, emb) if ghost else (None, None)
            curve.append({'update': u + 1, 'ghost_mass_to_mob': gm[0], 'mob_self_weight': gm[1], 'loss': round(float(loss), 5), 'inferred_per_step': round(float(acc.float().sum(1).mean()), 3),
                          'true_per_step': round(float(swap.float().sum(1).mean()), 3), 'precision': round(tp / max(tp + fp, 1), 3),
                          'recall': round(tp / max(int(swap.sum()), 1), 3), 'identity_weight_mean': round(float(w[..., 1].mean()), 3)})
    return curve


def main():
    import sys
    ghost = sys.argv[1:] == ['ghost']
    res = {}
    for rule in ('ORACLE', 'ML', 'MAP'):
        for seed in (0, 1):
            c = run(rule, seed, ghost)
            res[f'{rule}_s{seed}'] = c
            print(json.dumps({'rule': rule, 'seed': seed, 'start': c[0], 'early': c[1], 'mid': c[len(c) // 2], 'end': c[-1]}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'em_toy{"_ghost" if ghost else ""}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
