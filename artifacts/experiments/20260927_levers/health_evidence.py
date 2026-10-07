"""How much hit evidence does each representation hold at the TRAIN base rate? (2026-10-08, health diagnosis 3-4.)
A deterministic L1 head draws a hit only where P(hit | its input) > 0.5 (conditional median). On the health_chain subset,
probes are trained on balanced batches (fit episodes minus an inner split), early-stopped on inner AUC, then Platt-calibrated
on inner with POPULATION weights (health_strata.py: fresh hits x1, other hits x7.01, unchanged x361.4, i.e. the TRAIN mix at
t >= 14), and scored on held episodes:
  drawable   share of held hits with P > 0.5 (what an ideal deterministic head on these features could draw), per stratum
  false      share of held unchanged with P > 0.5
Reference feature sets (once): V = visible-history statistics (max zombie score beside the post-move cell with the TRUE scroll,
its > 0.3 flag, time since the last health drop one-hot, zombie beside in frames 11-14, current health one-hot); V without
`since`; the oracle ingredients alone (adj_post, adj_hist, since one-hot).
Per world (health_chain.py --ext file, window W = frames 15-W..14):
  drawable from h63 (the corrt head's only input for token 63), h_ext (h at 63, 31, the 12 cells within Manhattan 2, action
  token), and h63 + ORACLE ingredients (substitution: if supplying an ingredient lifts h63 to V's level, that ingredient is
  what h63 lacks): + since, + since + adj_post + adj_hist.
  ingredients (held AUC, uniform-weight probes) from h63 and h_ext: since == 6 (only if W >= 7: the drop 6 transitions back
  is inside the window), since in 1..4 (inside every window), adj_post (all, on scroll events, on scroll with a zombie within
  Manhattan 2 in frame 14).
  positive control (per W): the same probe on the TRUE token-63 history over the world's own window (frames 15-W..14).
  emitted: the world's own drawn hits / false drops (Probes.hud reader, 1.5 cut), per stratum.
Diagnosis 3 findings that motivated the substitution: 2026-10-08 NOTEBOOK. Writes evidence_<tag>.json.
Usage: health_evidence.py <tag> <world __ext.pt> ...
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def auc(s, y):
    s, y = s.cpu(), y.cpu()
    o = s.argsort(); _, c = torch.unique_consecutive(s[o], return_counts=True); e = c.cumsum(0).double()
    r = torch.empty(len(s), dtype=torch.float64); r[o] = ((e + e - c.double() + 1) / 2).repeat_interleave(c)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def weighted_probe(X, y, w, train, val, seed=0, steps=4000):
    """MLP trained on BALANCED batches (ranking; few positives otherwise starve a wide probe), early stop on validation AUC,
    then a weighted Platt fit P = sigmoid(a * logit + b) on the validation set with the population weights w, so that
    P is the population posterior and P > 0.5 means what a deterministic L1 head would draw. (2026-10-08: replaces the
    population-sampled training, which under-trained the 3,840-d h_ext probe: AUC 0.834 < h63's 0.898.)"""
    mu, sd = X[train].mean(0), X[train].std(0).clamp_min(1e-4)
    Z = ((X - mu) / sd).float().to(DEV)
    yd = y.to(DEV)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    pos, neg = torch.where(train & y)[0], torch.where(train & ~y)[0]
    g = torch.Generator().manual_seed(seed)
    best = (-1, None)
    for step in range(steps):
        b = torch.cat([pos[torch.randint(len(pos), (256,), generator=g)], neg[torch.randint(len(neg), (256,), generator=g)]]).to(DEV)
        loss = F.binary_cross_entropy_with_logits(net(Z[b])[:, 0], yd[b].float())
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 250 == 0:
            with torch.no_grad():
                v = auc(net(Z[val.to(DEV)])[:, 0], y[val])
            if v > best[0]:
                with torch.no_grad():
                    best = (v, net(Z)[:, 0].cpu())
    logit = best[1]
    a = torch.zeros(1, requires_grad=True); b0 = torch.zeros(1, requires_grad=True)
    cal = torch.optim.LBFGS([a, b0], max_iter=200)
    lv, yv, wv = logit[val], y[val].float(), w[val]
    def closure():
        cal.zero_grad()
        l = (F.binary_cross_entropy_with_logits(torch.exp(a) * lv + b0, yv, reduction='none') * wv).sum() / wv.sum()
        l.backward(); return l
    cal.step(closure)
    with torch.no_grad():
        return torch.sigmoid(torch.exp(a) * logit + b0)


def main():
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    st = torch.load(OUT / 'strata.pt', weights_only=False)
    cls = sub['classes']; hit, unch = cls == 1, cls == 2; keep = hit | unch
    N = len(cls); allm = torch.ones(N, dtype=torch.bool)
    fit = sub['fit']
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    held = ~fit
    w = st['weights']
    train, val = keep & fit & ~inner, keep & inner
    strata = {k: v for k, v in st['strata'].items()}
    strata['all'] = allm
    since = st['since']
    scroll = sub['scroll'] != 0
    near2 = sub['zombie_distance'] <= 2
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    # visible-history statistics V
    coords = torch.tensor([[c // 9, c % 9] for c in range(63)])
    disp = torch.tensor(SHIFTS)[sub['scroll']]
    d_post = (coords[None] - (torch.tensor([3, 4])[None] + disp)[:, None]).abs().sum(-1)
    zs = torch.zeros(N)
    for i in range(0, N, 512):
        z = P.zombie(torch.from_numpy(np.array(mm[i:i + 512, 14, :63])).float().flatten(0, 1))[:, 0].view(-1, 63)
        zs[i:i + 512] = torch.where(d_post[i:i + 512] == 1, z, torch.tensor(-9.)).amax(1)
    since1h = F.one_hot(since, 15).float()
    health1h = F.one_hot(sub['health'][:, 14].clamp(0, 9), 10).float()
    adj = st['adj_post'].float()[:, None]; adjh = st['adj_hist'].float()[:, None]
    oracle = torch.cat([adj, adjh, since1h], 1)
    V = torch.cat([zs[:, None], (zs > 0.3).float()[:, None], since1h, adjh, health1h], 1)
    V_no_since = torch.cat([zs[:, None], (zs > 0.3).float()[:, None], adjh, health1h], 1)
    raw63 = torch.from_numpy(np.array(mm[:, :15, 63])).float()                                  # [N,15,192] true history

    def drawable(Pr):
        draw = Pr > 0.5
        out = {k: [int((draw & hit & held & m).sum()), int((hit & held & m).sum())] for k, m in strata.items()}
        out['unchanged_false'] = [int((draw & unch & held).sum()), int((unch & held).sum())]
        out['held_auc_hit_vs_unchanged'] = round(auc(Pr[held & keep], hit[held & keep]), 4)
        return out
    probe = lambda X: drawable(weighted_probe(X, hit, w, train, val))

    def ingredients(X, W):
        targets = {'since_1to4': (since >= 1) & (since <= 4)}
        if W >= 7:
            targets['since_eq6'] = since == 6
        targets['adj_post'] = st['adj_post']
        out = {}
        for k, y in targets.items():
            pr = weighted_probe(X, y, torch.ones(N), allm & fit & ~inner, allm & inner, steps=2000)
            out[k] = round(auc(pr[held], y[held]), 4)
            if k == 'adj_post':
                out['adj_post|scroll'] = round(auc(pr[held & scroll], y[held & scroll]), 4)
                out['adj_post|scroll,zombie<=2'] = round(auc(pr[held & scroll & near2], y[held & scroll & near2]), 4)
        return out

    res = {'reference': {'V_visible_statistics': probe(V), 'V_without_since': probe(V_no_since), 'oracle_ingredients_only': probe(oracle)},
           'positive_control_true_token63_history': {}}
    print(json.dumps(res['reference']), flush=True)
    for path in sys.argv[2:]:
        r = torch.load(path, weights_only=False)
        W = r['window']
        if str(W) not in res['positive_control_true_token63_history']:
            res['positive_control_true_token63_history'][str(W)] = ingredients(raw63[:, 15 - W:].flatten(1), W)
            print(json.dumps({f'positive_control_w{W}': res['positive_control_true_token63_history'][str(W)]}), flush=True)
        he = r['h_ext'].float(); h63 = he[:, 0]
        wr = {'window': W,
              'drawable': {'h63': probe(h63), 'h_ext': probe(he.flatten(1)),
                           'h63+oracle_since': probe(torch.cat([h63, since1h], 1)),
                           'h63+oracle_since_adj': probe(torch.cat([h63, oracle], 1))},
              'ingredients_h63': ingredients(h63, W), 'ingredients_h_ext': ingredients(he.flatten(1), W)}
        # emitted (the world as trained), same strata, held
        read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
        cur = torch.cat([read(torch.from_numpy(np.array(mm[i:i + 2048, 14, 63:81]))) for i in range(0, N, 2048)])
        em = torch.cat([read(r['hud'][i:i + 2048]) for i in range(0, N, 2048)]) < cur - 1.5
        wr['emitted'] = {k: [int((em & hit & held & m).sum()), int((hit & held & m).sum())] for k, m in strata.items()}
        wr['emitted']['unchanged_false'] = [int((em & unch & held).sum()), int((unch & held).sum())]
        res[r['world'] + f'__w{W}'] = wr
        print(json.dumps({r['world'] + f'__w{W}': wr}), flush=True)
        (OUT / f'evidence_{sys.argv[1]}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
