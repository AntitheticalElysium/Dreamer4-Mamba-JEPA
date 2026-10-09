"""Does the stage-A posterior ENCODE the zombie's random move? (2026-10-09, written 12:10; E22FJ: the posterior's Delta never localizes
random zombie moves, 0.0% argmax on no-scroll steps, D 0.0%, while it removes 61% of scroll-entering error.)
Cases = health_zombie part 1 on steps WITHOUT scroll (player displacement 0, so the view does not shift): held diagnosis futures,
sample 0, exactly one zombie in view at t and t+1, alive, relative distance 1..6, move of length <= 1; the move class is stay / up /
down / left / right, kind toward / stay / other as health_zombie. For each case the posterior reads (true frame t, action t, true
frame t+1) as in training; features of the region (Posterior.encode's 5 x 5 blocks) holding the zombie's new cell:
  rel   the zombie's position relative to the post-move player: sign one-hots per axis and which axis is larger (9; the inputs of
        Craftax's toward rule, so rel predicts toward moves and cannot predict random ones)
  z     the normalized pre-quantization vector (64)          q   its codebook vector (64, what the decoder receives)
  z4    all four regions' z (256)                            z+rel
Probe: multinomial logistic regression (torch, standardized features, L2 1e-3, LBFGS), 5 folds grouped by walk seed; accuracy on all
cases and per kind. A posterior that encodes the observed move predicts `other` moves far above rel.
Runs on CPU. Usage: e22_zombiecode.py <name=dworld_a.pt> ...  -> artifacts/eda/health_chain_v1/e22_zombiecode.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, 'artifacts/experiments/20260927_levers')
import check_decision_step as CD  # noqa: E402
import dworld as DW  # noqa: E402
SD, T, H = CD.SD, CD.T, CD.H
OUT = Path('artifacts/eda/health_chain_v1')
SHIFTS = torch.tensor(((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
CENTER = torch.tensor([3, 4])
MOVES = {(0, 0): 0, (-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}


def cell_rc(mask):
    idx = mask.flatten(1).float().argmax(1)
    return torch.stack([idx // 9, idx % 9], 1)


def probe(x, y, groups, k=5, lam=1e-3):
    """grouped k-fold multinomial logistic regression -> out-of-fold predictions"""
    pred = torch.empty_like(y)
    ug = groups.unique()
    fold = {int(g): i % k for i, g in enumerate(ug[torch.randperm(len(ug), generator=torch.Generator().manual_seed(0))])}
    f = torch.tensor([fold[int(g)] for g in groups])
    for i in range(k):
        tr, te = f != i, f == i
        mu, sd = x[tr].mean(0), x[tr].std(0).clamp(min=1e-6)
        xt, xe = (x[tr] - mu) / sd, (x[te] - mu) / sd
        w = torch.zeros(x.shape[1], 5, requires_grad=True); b = torch.zeros(5, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(xt @ w + b, y[tr]) + lam * w.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        pred[te] = (xe @ w + b).argmax(1).detach()
    return pred


def main():
    from scroll import estimate
    meta, _, _ = T.split()
    fut5, _ = SD.token_cache(torch.device('cpu'))
    cache = T.build_cache('raw', torch.device('cpu'))
    R = len(meta['seed'])
    root, fa = cache['ctx'][:, -1], cache['fut_a']
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    zt = allv[..., 1071:1512].reshape(R, 17, 7, 9, 7)[..., 0] > 0
    frames = torch.cat([root[:, None], fut5[:, 0]], 1)                                            # [R, 17, 81, 192]
    sc = torch.cat([estimate(frames[i:i + 16, :-1].float().flatten(0, 1), frames[i:i + 16, 1:].float().flatten(0, 1)).view(-1, 16)
                    for i in range(0, R, 16)])
    n1 = zt.flatten(2).sum(-1)
    r_i, k_i = torch.where((n1[:, :-1] == 1) & (n1[:, 1:] == 1) & alive & (sc == 0))
    v0, v1 = cell_rc(zt[r_i, k_i]), cell_rc(zt[r_i, k_i + 1])
    rel0, move = v0 - CENTER, v1 - v0                                                             # no scroll: same view
    d0 = rel0.abs().sum(1)
    keep = (d0 >= 1) & (d0 <= 6) & (move.abs().sum(1) <= 1)
    r_i, k_i, v1, rel0, move = r_i[keep], k_i[keep], v1[keep], rel0[keep], move[keep]
    amax = rel0.abs().max(1, keepdim=True).values
    toward = (move.abs().sum(1) == 1) & ((move * -rel0.sign()) == 1).any(1) & ((move != 0) & (rel0.abs() == amax)).any(1)
    stay = move.abs().sum(1) == 0
    kind = torch.where(toward, 1, torch.where(stay, 0, 2))                                        # 0 stay, 1 toward, 2 other
    y = torch.tensor([MOVES[(int(a), int(b))] for a, b in move])
    groups = meta['seed'][r_i]
    sg = lambda v: F.one_hot(v.sign().long() + 1, 3).float()
    rel = torch.cat([sg(rel0[:, 0]), sg(rel0[:, 1]), F.one_hot((rel0[:, 0].abs() > rel0[:, 1].abs()).long()
                     + 2 * (rel0[:, 0].abs() == rel0[:, 1].abs()).long(), 3).float()], 1)
    region = 2 * (v1[:, 0] // 5) + (v1[:, 1] // 5)
    res = {'cases': {'stay': int((kind == 0).sum()), 'toward': int((kind == 1).sum()), 'other': int((kind == 2).sum())}}
    acc = lambda p, m: round(float((p[m] == y[m]).float().mean()), 3)
    p = probe(rel, y, groups)
    res['rel'] = {'all': acc(p, kind >= 0), **{kn: acc(p, kind == ki) for ki, kn in enumerate(('stay', 'toward', 'other'))}}
    print(json.dumps({'rel': res['rel'], 'cases': res['cases']}), flush=True)
    for name, path in (a.split('=', 1) for a in sys.argv[1:]):
        _, post, st = DW.load_a(Path(path), torch.device('cpu'))
        zs, qs = [], []
        with torch.no_grad():
            for i in range(0, len(r_i), 64):
                rr, kk = r_i[i:i + 64], k_i[i:i + 64]
                e = post.encode(frames[rr, kk].float(), fa[rr, kk], frames[rr, kk + 1].float())     # [n, 4, 1600]
                z = F.normalize(post.quantizer.pre(e), dim=-1)
                zs.append(z); qs.append(post.quantizer.codebook[(z @ post.quantizer.codebook.t()).argmax(-1)])
        z, q = torch.cat(zs), torch.cat(qs)                                                        # [n, 4, 64]
        ar = torch.arange(len(z))
        feats = {'z': z[ar, region], 'q': q[ar, region], 'z4': z.flatten(1), 'z+rel': torch.cat([z[ar, region], rel], 1)}
        out = {'world': st['name']}
        for fname, x in feats.items():
            p = probe(x, y, groups)
            out[fname] = {'all': acc(p, kind >= 0), **{kn: acc(p, kind == ki) for ki, kn in enumerate(('stay', 'toward', 'other'))}}
        out['codes_used_in_zombie_region'] = int((q[ar, region].unique(dim=0)).shape[0])
        res[name] = out
        print(json.dumps({name: out}), flush=True)
    (OUT / 'e22_zombiecode.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, 'artifacts/experiments/20260926_diagnosis')
    sys.path.insert(0, 'artifacts/experiments/20260921_readout_ladder')
    main()
