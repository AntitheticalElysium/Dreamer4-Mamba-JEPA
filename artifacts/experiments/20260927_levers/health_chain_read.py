"""Readouts for health_chain.py (2026-10-08). All on the fixed subset; probes / heads fit on `fit` episodes, scored on held ones.

Health is read by teval's HUD ridge (true-frame fit; health = hud[0] * 9) on 18 HUD tokens; a drawn hit = read(next) <
read(current true frame 14) - 1.5 (strict) or - 0.5 (lenient). Strata: hits (living dh <= -2), fresh hits (E20 flag), other hits
split scroll / stationary (true scroll 14 -> 15), unchanged (false drops), hard unchanged (a zombie within 2 cells in frame 14).
  controls   true frame 15 (must catch all), copy of frame 14 (must catch none)
  geometry   true token-63 change d = t15[63] - t14[63] per (health before, after): consistency of d (cosine to the group mean,
             relative residual), and the reader on copy + group-mean d (does a context-free delta depict the change?)
  info       held AUC of hit vs unchanged from (a) raw inputs (frame 14 tokens at 63, 31, the 12 cells within 2, + frames
             11-13 same cells, + action one-hot), (b) world h at 63, (c) h at 31 + 4 neighbours, (d) all six; linear (L2 logistic,
             strength by fit-internal episode validation) and MLP (256, early stop on the same validation)
  world      per world: drawn hits / false drops by stratum from its emitted HUD; corr mixture weights at 63; the generator
             candidate read alone (token 63 := generated); upstream on fresh hits: emitted scroll correct, zombie drawn beside
  refit      frozen world h -> a fresh linear token-63 generator, L1 on the true next token (class-balanced fit), read on held:
             what this backbone's h can express with a head trained only for it
Usage: health_chain_read.py <world result .pt> ...
"""
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
NEAR = [22, 30, 32, 40]
CELLS = [r * 9 + c for r in range(7) for c in range(9) if 0 < abs(r - 3) + abs(c - 4) <= 2]


def auc(s, y):
    o = s.argsort(); _, c = torch.unique_consecutive(s[o], return_counts=True); e = c.cumsum(0).double()
    r = torch.empty(len(s), dtype=torch.float64); r[o] = ((e + e - c.double() + 1) / 2).repeat_interleave(c)
    n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')


def probe(X, y, fit, inner, held, mlp=False, seed=0):
    """fit on fit & ~inner, choose L2 / stop on inner (episode-split part of fit), return held scores."""
    mu, sd = X[fit].mean(0), X[fit].std(0).clamp_min(1e-4)
    Z = ((X - mu) / sd).float()
    tr = fit & ~inner
    pos = float((~y[tr]).sum() / y[tr].sum().clamp_min(1))
    best = (-1, None)
    for lam in ((1e-4, 1e-3, 1e-2, 1e-1) if not mlp else (1e-4,)):
        torch.manual_seed(seed)
        net = (torch.nn.Sequential(torch.nn.Linear(Z.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 1)) if mlp
               else torch.nn.Linear(Z.shape[1], 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=lam if not mlp else 1e-2)
        idx = torch.where(tr)[0]
        g = torch.Generator().manual_seed(seed)
        for step in range(3000):
            b = idx[torch.randint(len(idx), (512,), generator=g)]
            loss = F.binary_cross_entropy_with_logits(net(Z[b])[:, 0], y[b].float(), pos_weight=torch.tensor(pos))
            if not mlp:
                loss = loss + lam * net.weight.square().sum()
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    v = auc(net(Z[inner])[:, 0], y[inner])
                if v > best[0]:
                    with torch.no_grad():
                        best = (v, net(Z)[:, 0].clone())
    return best[1]


def main():
    import teval as T
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    x = torch.from_numpy(np.asarray(np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))))
    meta, train_roots, train_seeds = T.split()
    P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, train_roots, train_seeds)
    read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
    cls, fresh, scroll, zd = sub['classes'], sub['fresh'], sub['scroll'], sub['zombie_distance']
    fit = sub['fit']
    import hashlib
    eps = torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    inner = fit & eps                                         # fit-internal validation (~1/4 of fit episodes), by episode
    held = ~fit
    hit, unch = cls == 1, cls == 2
    strata = {'hits': hit, 'fresh_hits': hit & fresh, 'scroll_hits_nonfresh': hit & ~fresh & (scroll != 0),
              'stationary_hits': hit & (scroll == 0), 'unchanged': unch, 'hard_unchanged': unch & (zd <= 2),
              'stationary_unchanged_zombie_beside': unch & (zd == 1) & (scroll == 0), 'deaths': cls == 0}
    t14, t15 = x[:, 14, 63:81], x[:, 15, 63:81]
    cur = torch.cat([read(t14[i:i + 2048]) for i in range(0, len(x), 2048)])
    def drawn(hud, cut):
        r = torch.cat([read(hud[i:i + 2048]) for i in range(0, len(x), 2048)])
        return r < cur - cut, r
    def table(hud, where=None):
        out = {}
        for cut in (1.5, 0.5):
            d, r = drawn(hud, cut)
            out[str(cut)] = {k: [int(d[m & where].sum()), int((m & where).sum())] if where is not None else [int(d[m].sum()), int(m.sum())]
                             for k, m in strata.items()}
        return out
    res = {'strata_counts': {k: int(m.sum()) for k, m in strata.items()},
           'held_counts': {k: int((m & held).sum()) for k, m in strata.items()},
           'controls': {'true_next': table(t15), 'copy': table(t14)}}
    # geometry of the true token-63 change
    h0, h1 = sub['health'][:, 14], sub['health'][:, 15]
    d = (x[:, 15, 63] - x[:, 14, 63]).float()
    geo = {}
    mean_d = {}
    for a, b in sorted({(int(p), int(q)) for p, q in zip(h0[hit], h1[hit])}):
        m = hit & (h0 == a) & (h1 == b) & fit
        if int(m.sum()) < 20:
            continue
        md = d[m].mean(0); mean_d[(a, b)] = md
        mh = hit & (h0 == a) & (h1 == b) & held
        if int(mh.sum()) == 0:
            continue
        cos = F.cosine_similarity(d[mh], md[None], dim=-1)
        rel = (d[mh] - md).norm(dim=-1) / d[mh].norm(dim=-1)
        hud = t14[mh].clone(); hud[:, 0] = (x[mh, 14, 63].float() + md).half()
        r = read(hud)
        geo[f'{a}->{b}'] = {'n_held': int(mh.sum()), 'cos_to_mean_median': float(cos.median()),
                            'relative_residual_median': float(rel.median()), 'change_norm_median': float(d[mh].norm(dim=-1).median()),
                            'copy_plus_mean_delta_drawn_1.5': float((r < cur[mh] - 1.5).float().mean()),
                            'copy_plus_mean_delta_read_mean': float(r.mean())}
    res['geometry'] = geo
    # information: raw inputs
    keep = hit | unch
    y = hit
    act = F.one_hot(sub['actions'][:, 14], 17).float()
    raw = torch.cat([x[:, 11:15][:, :, [63, 31] + CELLS].float().flatten(1), act], 1)
    pairs = (('hits', 'unchanged'), ('fresh_hits', 'unchanged'), ('fresh_hits', 'hard_unchanged'),
             ('stationary_hits', 'stationary_unchanged_zombie_beside'), ('scroll_hits_nonfresh', 'unchanged'))
    def info(X, mlp_too=True):
        out = {}
        for kind, mlp in (('linear', False), ('mlp', True))[:2 if mlp_too else 1]:
            s = probe(X[keep], y[keep], fit[keep], inner[keep], held[keep], mlp=mlp)
            sk = torch.full((len(x),), float('nan')); sk[keep] = s
            out[kind] = {}
            for k, neg in pairs:
                m = held & (strata[k] | strata[neg])
                out[kind][f'{k}_vs_{neg}'] = [round(auc(sk[m], strata[k][m]), 4), int((held & strata[k]).sum()), int((held & strata[neg]).sum())]
        return out
    res['info_raw_inputs'] = info(raw)
    for path in sys.argv[1:]:
        r = torch.load(path, weights_only=False)
        name = r['world'] + f"__w{r['window']}"
        wres = {'emitted': table(r['hud'])}
        gh = r['hud'].clone(); gh[:, 0] = r['gen63']
        wres['generator_alone'] = table(gh)
        w = r['weights63']
        wres['weights63_mean'] = {k: [round(float(v), 4) for v in w[m].mean(0)] for k, m in strata.items()}
        z_true = (P.zombie(x[:, 15, NEAR].float().flatten(0, 1))[:, 0].view(-1, 4) > 0.3).any(1)
        z_pred = (P.zombie(r['near_out'].float().flatten(0, 1))[:, 0].view(-1, 4) > 0.3).any(1)
        fh = strata['fresh_hits']
        wres['upstream_fresh_hits'] = {'n': int(fh.sum()), 'scroll_correct': float((r['pred_scroll'][fh] == scroll[fh]).float().mean()),
                                       'zombie_beside_true': float(z_true[fh].float().mean()),
                                       'zombie_beside_drawn': float(z_pred[fh].float().mean()),
                                       'zombie_beside_drawn_given_true': float(z_pred[fh & z_true].float().mean())}
        h = r['h'].float()
        wres['info_h63'] = info(h[:, 0])
        wres['info_h_player_near'] = info(h[:, 1:].flatten(1), mlp_too=False)
        wres['info_h_all'] = info(h.flatten(1))
        # refit: fresh linear token-63 generator from frozen h (all six positions), class-balanced L1 on fit
        X = h.flatten(1); mu, sd = X[fit].mean(0), X[fit].std(0).clamp_min(1e-4); Z = (X - mu) / sd
        tgt = x[:, 15, 63].float()
        torch.manual_seed(0); lin = torch.nn.Linear(Z.shape[1], 192)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3, weight_decay=1e-2)
        g = torch.Generator().manual_seed(0)
        pools = [torch.where(fit & hit & ~inner)[0], torch.where(fit & unch & ~inner)[0]]
        best = (1e9, None)
        for step in range(4000):
            b = torch.cat([p[torch.randint(len(p), (256,), generator=g)] for p in pools])
            loss = (lin(Z[b]) - tgt[b]).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    v = (lin(Z[inner & (hit | unch)]) - tgt[inner & (hit | unch)]).abs().mean().item()
                if v < best[0]:
                    best = (v, {k: t.clone() for k, t in lin.state_dict().items()})
        lin.load_state_dict(best[1])
        with torch.no_grad():
            rh = r['hud'].clone(); rh[:, 0] = F.layer_norm(lin(Z), (192,)).half()
        wres['refit_linear_from_h'] = {'held_only': True, **table(rh, held)}
        wres['emitted_held'] = table(r['hud'], held)
        res[name] = wres
        print(json.dumps({name: {k: v for k, v in wres.items() if k in ('emitted', 'upstream_fresh_hits', 'weights63_mean')}}), flush=True)
    (OUT / 'readout.json').write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({'controls': res['controls'], 'geometry': res['geometry'], 'info_raw_inputs': res['info_raw_inputs']}), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
