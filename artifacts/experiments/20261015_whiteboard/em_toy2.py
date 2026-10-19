"""Synthetic test 2 (2026-10-15, CPU): global prior vs the model's own context-dependent prior in inferred-chance training.

Same ring world as em_toy.py, in a HARD regime (token noise 0.3, mobs barely distinct: scale 1.0), where the evidence for
genuine moves (gain / s) often falls near the MDL threshold, as for the real D-parent afterstate (MAP recall 54%). Both
arms train the afterstate through T_hat (point loss) AND a chance head c(x_j, x_{j+1}) -> logit of swapping pair (j, j+1),
by binary cross-entropy toward the soft posterior q:
  GLOBAL   q = sigmoid(gain/s + logit pi), pi = running posterior mass per pair; T_hat = q > 0.5     (TRANSPORT_IC v2)
  CONTEXT  q = sigmoid(gain/s + c(x)): the chance head's own prior for that pair; T_hat = q > 0.5   (EM with model prior)
  ORACLE   T_hat = truth, chance head trained on the truth
Readings: posterior recall / precision vs truth, inferred and sampled (Bernoulli(sigmoid(c))) swaps per step vs true, the
chance head's calibration at mob pairs (mean predicted swap probability vs true frequency) and at terrain pairs.
Usage: em_toy2.py -> artifacts/eda/whiteboard/em_toy2.json
"""
import json
import math
from pathlib import Path

import torch

import em_toy as E

OUT = E.OUT


class Chance(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(2 * E.D, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        with torch.no_grad():
            self.f[-1].bias.fill_(-4.0)

    def forward(self, x):
        return self.f(torch.cat([x, x.roll(-1, 1)], -1))[..., 0]          # [B,N] logit of swapping (j, j+1)


def world_hard(g, emb):
    x, z, drift, swap = E.world(g, emb)
    return x + .25 * torch.randn(x.shape, generator=g), z + .25 * torch.randn(z.shape, generator=g), drift, swap


def posterior(a, z, prior_logit):
    k_idx = torch.arange(E.N).roll(-1)
    res = (a - z).norm(dim=-1)
    gain = res + res[:, k_idx] - (a[:, k_idx] - z).norm(dim=-1) - (a - z[:, k_idx]).norm(dim=-1)
    s = res.mean(1, keepdim=True)
    q = torch.sigmoid(gain / s + prior_logit)
    acc = torch.zeros_like(gain, dtype=torch.bool)                        # greedy one-to-one over posterior odds > 1
    for b in range(len(a)):
        used = set()
        for j in torch.argsort(q[b], descending=True).tolist():
            if q[b, j] <= .5:
                break
            k = (j + 1) % E.N
            if j in used or k in used:
                continue
            used |= {j, k}; acc[b, j] = True
    return acc, q                                                           # soft targets: the posterior marginal of every pair


def run(rule, seed):
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(4, E.D, generator=g)
    torch.manual_seed(seed)
    m, c = E.Model(), Chance()
    opt = torch.optim.Adam(list(m.parameters()) + list(c.parameters()), lr=3e-3)
    pi = .02; curve = []
    for u in range(3000):
        x, z, drift, swap = world_hard(g, emb)
        a, _ = m(x, drift)
        logit = c(x)
        with torch.no_grad():
            if rule == 'ORACLE':
                acc, q = swap, swap.float()
            elif rule == 'GLOBAL':
                acc, q = posterior(a.detach(), z, math.log(pi / (1 - pi)))
                pi = min(max(.99 * pi + .01 * float(q.sum(1).mean()) / E.N, 1e-4), .5)
            else:
                acc, q = posterior(a.detach(), z, logit.detach())
        loss = (E.apply(a, acc) - z).square().mean() + torch.nn.functional.binary_cross_entropy_with_logits(logit, q)
        opt.zero_grad(); loss.backward(); opt.step()
        if (u + 1) % 500 == 0:
            with torch.no_grad():
                xm, _, _, sw = world_hard(g, emb)
                p = torch.sigmoid(c(xm))
                # mob pairs: pairs touching a mob cell (by the clean class: distance to the mob embedding)
                mobcell = ((xm[..., None, :] - emb[None, None]).norm(dim=-1).argmin(-1) == 3)
                mobpair = mobcell | mobcell.roll(-1, 1)
            tp = int((acc & swap).sum()); fp = int((acc & ~swap).sum())
            curve.append({'update': u + 1, 'recall': round(tp / max(int(swap.sum()), 1), 3), 'precision': round(tp / max(tp + fp, 1), 3),
                          'inferred_per_step': round(float(acc.float().sum(1).mean()), 3), 'true_per_step': round(float(swap.float().sum(1).mean()), 3),
                          'sampled_per_step': round(float(p.sum(1).mean()), 3),
                          'head_p_at_mob_pairs': round(float(p[mobpair].mean()), 4), 'true_freq_at_mob_pairs': round(float(sw[mobpair].float().mean()), 4),
                          'head_p_at_terrain_pairs': round(float(p[~mobpair].mean()), 5)})
    return curve


def main():
    res = {}
    for rule in ('ORACLE', 'GLOBAL', 'CONTEXT'):
        for seed in (0, 1):
            cv = run(rule, seed)
            res[f'{rule}_s{seed}'] = cv
            print(json.dumps({'rule': rule, 'seed': seed, 'u500': cv[0], 'u3000': cv[-1]}), flush=True)
    (OUT / 'em_toy2.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
