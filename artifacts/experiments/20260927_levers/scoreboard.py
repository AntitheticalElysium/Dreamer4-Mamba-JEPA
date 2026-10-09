"""10-10 PROGRAM Part I (predeclared 09:10): the breadth scoreboard, read from rollout_export.py's saved predictions. CPU.
Truth: teval's Raw diagnosis cache (sample 0 = the factual future; teval's V) and stochdiag's five futures / four one-step keys, with
the simulator's visible state of every true frame. Judged on teval's TEST seeds where a probe is involved; errors on all roots.
Per variant (artifacts/eda/scoreboard_v1/<variant>/scoreboard.json, per-root rows in rows.pt):
  errors      teval's: gen_k / V and tf_k / V for k = 1..16 (alive sample 0), one-step error / copy per transition class
              (onestep.classify)
  decomposition (stochdiag's definitions; roots with all five futures alive at k; classes hud / player / mob / entering / static):
              floor (the unbiased variance of the five true futures = the best deterministic world's error), excess
              (|g - mu|^2 - floor / 5), both / V per root; energy score per class (proper for distributions: mean_s |x - y_s| for one
              prediction; for a group of samples mean_{j,s} |x_j - y_s| - 0.5 mean_{j != l} |x_j - x_l|), |.| over the class's tokens
  facts       ridge probes fitted on TRUE TRAIN-seed tokens (root + factual future; lambda on a held-out fifth, teval.ridge): tile
              class (62 map cells), zombie / skeleton / cow / arrow per cell, health / food / drink / energy and 12 inventory items
              (18 HUD tokens), facing and sleep (player token), light (mean map token); read on TEST roots at depths 1, 2, 4, 8, 16 for
              gen, tf, copy-root and the true tokens (ceiling): tile accuracy (all / near / interior / edge), mob AUCs, mob census
              (cells above a per-type threshold set on TRAIN true tokens to reproduce the true mean count) vs sample 0 and the mean
              of the five futures, HUD R^2 and bias, inventory R^2, facing / sleep accuracy, light R^2
  scroll      scroll.estimate on consecutive frames, imagined vs true: share equal, missed (true moves, imagined does not), false,
              wrong direction, by depth; aligned share (cumulative offsets equal, driftanat.offsets) by depth
  hud_events  per step, imagined decrement (readout falls by > 0.5) rate of health / food / drink / energy vs the five real futures'
              rate on the same steps (in distribution: a sampled world should match the RATE, not the realized step)
  damage      e22_imagine's readings exactly (teval Probes; caught all / aligned, false, beside rate, hit rate given beside, zombie
              census, one-zombie localization) -- for the E22MS samples they must reproduce e22_imagine's JSON
  manifold    squared distance to the nearest of the 4,096 k-means codes of true training tokens (stochdiag.nearest), per class and
              depth, imagined vs the true frame
--group NAME=v1,v2,... scores those variants together as one sampled world (energy score; mean of the per-sample readings).
Usage: scoreboard.py <variant> ... [--group NAME=v1,v2,..]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
H, N, S = 16, 17, 5
DEPTHS = (1, 2, 4, 8, 16)
MAPC = [c for c in range(63) if c != 31]
NEAR4 = [22, 30, 32, 40]
EDGE = [r * 9 + c for r in range(7) for c in range(9) if (r in (0, 6) or c in (0, 8))]
INTER = [c for c in MAPC if c not in EDGE and c not in NEAR4]
MOBS = ('zombie', 'skeleton', 'cow', 'arrow')


def auc(score, label):
    from compound import auc as a
    return a(score, label)


class AllProbes:
    def __init__(self, T, cache, meta, train_roots, train_seeds):
        seeds = meta['seed']
        val = torch.isin(seeds, train_seeds[: len(train_seeds) // 5])
        fit = train_roots & ~val
        t = torch.cat([cache['ctx'][:, -1:], cache['fut']], 1).flatten(0, 1)                    # [R*17,81,192] fp16
        v = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).flatten(0, 1).float()
        rows = lambda m: m[:, None].expand(-1, 17).flatten()
        fr, vr = rows(fit), rows(val)
        f = self.facts(v)
        cr = lambda m: m[:, None].expand(-1, 62).flatten()
        ncell = len(t) * 62
        sub = torch.randperm(ncell, generator=torch.Generator().manual_seed(0))[:400_000]
        sub = sub[cr(fr)[sub] | cr(vr)[sub]]
        cell = t[sub // 62, torch.tensor(MAPC)[sub % 62]].float()                                   # only the sampled cells
        kf, kv = cr(fr)[sub], cr(vr)[sub]
        tile_y, mob_y = f['tile'].flatten()[sub], f['mobs'].flatten(0, 1)[sub]
        self.tile = T.ridge(cell, F.one_hot(tile_y, 17).float(), kf, kv)
        self.mob = T.ridge(cell, mob_y.float(), kf, kv)
        hud_x = t[:, 63:81].flatten(1).float()
        self.hud = T.ridge(hud_x, f['hud'], fr, vr)
        del hud_x
        pl = t[:, 31].float()
        self.facing = T.ridge(pl, F.one_hot(f['facing'], 4).float(), fr, vr)
        self.sleep = T.ridge(pl, f['sleep'][:, None], fr, vr)
        self.light = T.ridge(torch.cat([t[i:i + 2048][:, MAPC].float().mean(1) for i in range(0, len(t), 2048)]), f['light'][:, None], fr, vr)
        # per-type thresholds reproducing the true mean count on TRAIN true tokens (the sampled fit cells)
        sc = self.mob(cell[kf]); lab = mob_y[kf]
        self.thr = []
        for j in range(4):
            n_true = int(lab[:, j].sum())
            self.thr.append(float(sc[:, j].sort(descending=True).values[max(n_true - 1, 0)]) if n_true else 0.5)

    @staticmethod
    def facts(v):
        v = v.float()
        m = v[..., 1071:1512].reshape(*v.shape[:-1], 63, 7)[..., MAPC, :] > 0
        return {'tile': v[..., :1071].reshape(*v.shape[:-1], 63, 17)[..., MAPC, :].argmax(-1),
                'mobs': torch.stack([m[..., 0], m[..., 2], m[..., 1], m[..., 3:].any(-1)], -1),
                'hud': torch.cat([v[..., 1512:1516], v[..., 1522:1534]], -1) * 9,
                'facing': v[..., 1516:1520].argmax(-1), 'sleep': v[..., 1520], 'light': v[..., 1521]}

    def read(self, tok, vis):
        """tok [n,81,192], vis [n,1534] -> dict of fact metrics"""
        tok = tok.float(); f = self.facts(vis)
        n = len(tok)
        tp = self.tile(tok[:, MAPC].flatten(0, 1)).view(n, 62, 17).argmax(-1)
        ok = (tp == f['tile']).float()
        idx = {c: i for i, c in enumerate(MAPC)}
        reg = lambda cells: round(float(ok[:, [idx[c] for c in cells]].mean()), 4)
        ms = self.mob(tok[:, MAPC].flatten(0, 1)).view(n, 62, 4)
        out = {'tile_all': round(float(ok.mean()), 4), 'tile_near': reg(NEAR4), 'tile_interior': reg(INTER), 'tile_edge': reg(EDGE)}
        for j, nm in enumerate(MOBS):
            lab = f['mobs'][..., j]
            out[f'{nm}_auc'] = round(auc(ms[..., j].flatten(), lab.flatten()), 4) if 0 < lab.sum() < lab.numel() else None
            out[f'{nm}_count'] = round(float((ms[..., j] > self.thr[j]).float().sum(1).mean()), 4)
            out[f'{nm}_count_true'] = round(float(lab.float().sum(1).mean()), 4)
        hp = self.hud(tok[:, 63:81].flatten(1))
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum().clamp_min(1e-9))
        for j, nm in enumerate(('health', 'food', 'drink', 'energy')):
            out[f'{nm}_r2'] = round(r2(hp[:, j], f['hud'][:, j]), 4); out[f'{nm}_bias'] = round(float((hp[:, j] - f['hud'][:, j]).mean()), 4)
        inv = [r2(hp[:, 4 + j], f['hud'][:, 4 + j]) for j in range(12) if f['hud'][:, 4 + j].var() > 0]
        out['inventory_r2_mean'] = round(sum(inv) / max(len(inv), 1), 4)
        out['facing_acc'] = round(float((self.facing(tok[:, 31]).argmax(-1) == f['facing']).float().mean()), 4)
        out['sleep_acc'] = round(float(((self.sleep(tok[:, 31])[:, 0] > 0.5) == (f['sleep'] > 0.5)).float().mean()), 4)
        out['light_r2'] = round(r2(self.light(tok[:, MAPC].mean(1))[:, 0], f['light']), 4)
        return out


def energy(X, Y, m):
    """X [b,J,81,192] predictions, Y [b,S,81,192] truths, m [b,81] class mask -> energy score per root [b]"""
    dxy = torch.stack([torch.stack([(((X[:, j] - Y[:, s]) ** 2).sum(-1) * m).sum(-1).sqrt() for s in range(Y.shape[1])], 1)
                       for j in range(X.shape[1])], 1)                                                 # [b,J,S]
    es = dxy.mean((1, 2))
    J = X.shape[1]
    if J > 1:
        dxx = torch.stack([(((X[:, j] - X[:, l]) ** 2).sum(-1) * m).sum(-1).sqrt() for j in range(J) for l in range(J) if j != l], 1)
        es = es - 0.5 * dxx.mean(1)
    return es


def main():
    import teval as T
    import stochdiag as SD
    import driftanat as DA
    from onestep import CLASSES as OCL, classify
    from scroll import estimate
    torch.set_num_threads(8)
    args = sys.argv[1:]
    groups = dict(a.split('=', 1) for a in (args[i + 1] for i, x in enumerate(args) if x == '--group'))
    groups = {k: v.split(',') for k, v in groups.items()}
    variants = [a for i, a in enumerate(args) if a != '--group' and (i == 0 or args[i - 1] != '--group')]
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5, one4 = SD.token_cache(torch.device('cpu'))
    R = len(meta['seed'])
    fut, root = cache['fut'], cache['ctx'][:, -1].float()                                          # fut fp16; converted per chunk
    fmean = torch.stack([fut[i:i + 64].float().flatten(0, 1).sum(0) for i in range(0, R, 64)]).sum(0) / (R * H)
    V = float(sum(((fut[i:i + 64].float() - fmean) ** 2).sum((-1, -2)).sum() for i in range(0, R, 64)) / (R * H))
    alive0 = ~meta['future_dead'][:, 0].cumsum(1).bool()                                             # sample 0
    alive5 = ~meta['future_dead'].cumsum(2).bool().any(1)                                            # all five
    ocls, _ = classify(meta)
    cls, _ = SD.classes(meta, fut5[:, 0], cache['ctx'][:, -1])
    probes = AllProbes(T, cache, meta, train_roots, train_seeds)
    tprobes = T.Probes(cache, meta, train_roots, train_seeds)                                        # e22_imagine's reader
    codes = torch.load(ROOT / 'artifacts/eda/levers_codebooks_v1/raw_K4096.pt', weights_only=False)['codes'].float()
    vis5 = meta['future_visible']                                                                    # [R,5,16,1534] fp16
    true_sh = torch.cat([estimate(torch.cat([cache['ctx'][i:i + 32, -1:], cache['fut'][i:i + 32, :-1]], 1).float().flatten(0, 1),
                                  fut[i:i + 32].float().flatten(0, 1)).view(-1, H) for i in range(0, R, 32)])
    true_off = DA.offsets(true_sh)
    # truth for e22_imagine's damage readings (its masks exactly)
    allv = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1).float()
    hp = allv[:, :, 1512] * 9
    drop = (hp[:, 1:] < hp[:, :-1] - 0.5) & alive5
    same = (hp[:, 1:] == hp[:, :-1]) & alive5
    zt = allv[..., 1071:1512].reshape(R, 17, 7, 9, 7)[..., 0] > 0
    n1 = zt.flatten(2).sum(-1)
    true_beside = zt[:, :-1].flatten(2)[:, :, NEAR4].any(-1) & alive5
    zscore_t = torch.stack([tprobes.zombie(fut5[:, 0, k, :63].float().flatten(0, 1))[:, 0].view(R, 7, 9) for k in range(H)], 1)
    true_peak = zscore_t.flatten(2).max(2).values
    # real futures' HUD decrement rates per step (all five; in-distribution reference)
    hud5 = torch.cat([meta['root_visible'][:, None, None].expand(-1, S, 1, -1), meta['future_visible']], 2).float()[..., 1512:1516] * 9
    dec5 = (hud5[:, :, 1:] < hud5[:, :, :-1] - 0.5)                                                  # [R,5,16,4]

    def load(name):
        d = OUT / name
        g = lambda n: torch.from_numpy(np.load(d / f'{n}.npy', mmap_mode='r'))
        return {'gen': g('gen'), 'tf': g('tf'), 'one': g('one'), 'meta': json.loads((d / 'meta.json').read_text())}

    def score(names, label):
        vs = [load(n) for n in names]
        J = len(vs)
        res = {'variants': names, 'V': V}
        rows = {}
        # errors (teval's)
        ge = torch.stack([torch.cat([((v['gen'][i:i + 64].float() - fut[i:i + 64].float()) ** 2).sum((-1, -2)) for i in range(0, R, 64)]) for v in vs])   # [J,R,16]
        te = torch.stack([torch.cat([((v['tf'][i:i + 64].float() - fut[i:i + 64].float()) ** 2).sum((-1, -2)) for i in range(0, R, 64)]) for v in vs])
        rows['gen_err'], rows['tf_err'] = ge, te
        res['errors'] = {'gen_over_V': [round(float(ge[:, alive0[:, k], k].mean() / V), 5) for k in range(H)],
                         'tf_over_V': [round(float(te[:, alive0[:, k], k].mean() / V), 5) for k in range(H)]}
        true_one = cache['one']
        oe = torch.stack([torch.cat([((v['one'][i:i + 64].float() - true_one[i:i + 64].float()) ** 2).sum((-1, -2)) for i in range(0, R, 64)]) for v in vs])
        cp = torch.cat([((root[i:i + 64, None] - true_one[i:i + 64].float()) ** 2).sum((-1, -2)) for i in range(0, R, 64)])
        res['errors']['onestep_over_copy'] = {c: round(float(oe[:, ocls == i].mean() / cp[ocls == i].mean()), 4) for i, c in enumerate(OCL)}
        res['errors']['onestep_over_copy']['all'] = round(float(oe.mean() / cp.mean()), 4)
        # decomposition + energy score
        dec = {k: {c: {'floor': 0.0, 'excess': 0.0, 'es': 0.0} for c in SD.CLASSES} for k in DEPTHS}
        es_rows = torch.full((R, len(DEPTHS), len(SD.CLASSES)), float('nan'))
        nk = {k: int(alive5[:, k - 1].sum()) for k in DEPTHS}
        for i in range(0, R, 16):
            b = min(16, R - i)
            Y = fut5[i:i + b].float()                                                                   # [b,5,16,81,192]
            mu = Y.mean(1)
            s2 = ((Y - mu[:, None]) ** 2).sum((1, -1)) / (S - 1)                                        # [b,16,81]
            for di, k in enumerate(DEPTHS):
                X = torch.stack([v['gen'][i:i + b, k - 1].float() for v in vs], 1)                    # [b,J,81,192]
                m_root = alive5[i:i + b, k - 1]
                exc = ((X - mu[:, None, k - 1]) ** 2).sum(-1).mean(1) - s2[:, k - 1] / S             # [b,81] (mean over samples)
                for ci, c in enumerate(SD.CLASSES):
                    m = (cls[i:i + b, k - 1] == ci) & m_root[:, None]
                    dec[k][c]['floor'] += float(s2[:, k - 1][m].sum()); dec[k][c]['excess'] += float(exc[m].sum())
                    es = energy(X, Y[:, :, k - 1], ((cls[i:i + b, k - 1] == ci)).float())
                    dec[k][c]['es'] += float(es[m_root].sum())
                    es_rows[i:i + b, di, ci] = torch.where(m_root, es, torch.tensor(float('nan')))
        rows['es'] = es_rows
        res['decomposition'] = {str(k): {c: {q: round(dec[k][c][q] / max(nk[k], 1) / (V if q != 'es' else V ** 0.5), 5)
                                             for q in ('floor', 'excess', 'es')} for c in SD.CLASSES} for k in DEPTHS}
        for k in DEPTHS:
            d_ = res['decomposition'][str(k)]
            d_['all'] = {q: round(sum(d_[c][q] for c in SD.CLASSES), 5) for q in ('floor', 'excess')}
        # facts by depth (TEST roots), per sample then mean over samples
        res['facts'] = {}
        for k in DEPTHS:
            m = alive0[:, k - 1] & test
            vis = meta['future_visible'][:, 0, k - 1][m]
            per = [probes.read(v['gen'][m, k - 1], vis) for v in vs]
            row = {key: (round(sum(p[key] for p in per) / J, 4) if per[0][key] is not None else None) for key in per[0]}
            row_tf = probes.read(vs[0]['tf'][m, k - 1], vis)
            row_true = probes.read(fut[m, k - 1].float(), vis)
            row_copy = probes.read(root[m], vis)
            five = vis5[m][:, :, k - 1].float()                                                          # [n,5,1534]
            for j, nm in enumerate(MOBS):
                row[f'{nm}_count_5futures'] = round(float(AllProbes.facts(five)['mobs'][..., j].float().sum(-1).mean()), 4)
            res['facts'][str(k)] = {'imagined': row, 'teacher_forced': row_tf, 'true_tokens': row_true, 'copy_root': row_copy}
        # scroll fidelity and alignment
        sc = {}
        al = []
        for v in vs:
            gsh = torch.cat([estimate(torch.cat([cache['ctx'][i:i + 32, -1:].float(), v['gen'][i:i + 32, :-1].float()], 1).flatten(0, 1),
                                      v['gen'][i:i + 32].float().flatten(0, 1)).view(-1, H) for i in range(0, R, 32)])
            al.append((DA.offsets(gsh) == true_off).all(-1))
            for k in range(H):
                mk = alive0[:, k]
                ts, is_ = true_sh[mk, k], gsh[mk, k]
                acc = sc.setdefault(k + 1, [0.0, 0.0, 0.0, 0.0, 0, 0, 0])
                acc[0] += float((ts == is_).float().sum()); acc[1] += float(((ts != 0) & (is_ == 0)).float().sum())
                acc[2] += float(((ts == 0) & (is_ != 0)).float().sum()); acc[3] += float(((ts != 0) & (is_ != 0) & (ts != is_)).float().sum())
                acc[4] += int(mk.sum()); acc[5] += int(((ts != 0)).sum()); acc[6] += int((ts == 0).sum())
        aligned = torch.stack(al).float().mean(0)                                                        # [R,16]
        rows['aligned'] = aligned
        res['scroll'] = {str(k): {'equal': round(a[0] / a[4], 4), 'missed_given_true_move': round(a[1] / max(a[5], 1), 4),
                                  'false_given_true_still': round(a[2] / max(a[6], 1), 4),
                                  'wrong_direction_given_true_move': round(a[3] / max(a[5], 1), 4),
                                  'aligned': round(float(aligned[alive0[:, k - 1], k - 1].mean()), 4)} for k, a in sc.items() if k in DEPTHS}
        # HUD decrement events in distribution
        ev = {}
        for j, nm in enumerate(('health', 'food', 'drink', 'energy')):
            rates = []
            for v in vs:
                hr = torch.cat([probes.hud(torch.cat([cache['ctx'][i:i + 32, -1:].float(), v['gen'][i:i + 32].float()], 1)[:, :, 63:81].flatten(2).flatten(0, 1))[:, j].view(-1, H + 1)
                                for i in range(0, R, 32)])                                               # [R,17]
                rates.append((hr[:, 1:] < hr[:, :-1] - 0.5)[alive5].float().mean())
            ev[nm] = {'imagined_rate': round(float(torch.stack(rates).mean()), 5),
                      'real_futures_rate': round(float(dec5[..., j].permute(0, 2, 1)[alive5].float().mean()), 5)}
        res['hud_events'] = ev
        # damage (e22_imagine's readings exactly), per sample, then mean
        dmg = []
        for v in vs:
            g = v['gen']
            pred_h = torch.stack([tprobes.hud(g[:, k, 63:81].float().flatten(1))[:, 0] * 9 for k in range(H)], 1)
            cur_h = torch.cat([tprobes.hud(cache['ctx'][:, -1, 63:81].float().flatten(1))[:, :1] * 9,
                               torch.stack([tprobes.hud(g[:, k, 63:81].float().flatten(1))[:, 0] * 9 for k in range(H - 1)], 1)], 1)
            znear = torch.stack([(tprobes.zombie((cache['ctx'][:, -1] if k == 0 else g[:, k - 1])[:, NEAR4].float().flatten(0, 1))[:, 0].view(R, 4) > 0.3).any(1)
                                 for k in range(H)], 1)
            zs = torch.stack([tprobes.zombie(g[:, k, :63].float().flatten(0, 1))[:, 0].view(R, 7, 9) for k in range(H)], 1)
            peak = zs.flatten(2).max(2).values
            gprev = torch.cat([cache['ctx'][:, -1:].float(), g[:, :-1].float()], 1)
            off = DA.offsets(torch.cat([estimate(gprev[j:j + 32].flatten(0, 1), g[j:j + 32].float().flatten(0, 1)).view(-1, H) for j in range(0, R, 32)]))
            tr_prev = torch.cat([cache['ctx'][:, -1:], fut5[:, 0, :-1]], 1).float()
            t_off = DA.offsets(torch.cat([estimate(tr_prev[j:j + 16].flatten(0, 1), fut5[j:j + 16, 0].float().flatten(0, 1)).view(-1, H) for j in range(0, R, 16)]))
            al_ = torch.cat([torch.ones(R, 1, dtype=torch.bool), (off == t_off).all(-1)[:, :-1]], 1)
            pdrop = pred_h < cur_h - 0.5
            r = {'caught_all': round(float(pdrop[drop].float().mean()), 4),
                 'caught_aligned': round(float(pdrop[drop & al_].float().mean()), 4),
                 'false_all': round(float(pdrop[same].float().mean()), 4),
                 'beside_rate': round(float(znear[alive5].float().mean()), 4), 'true_beside_rate': round(float(true_beside[alive5].float().mean()), 4),
                 'hit_rate_given_beside': round(float(pdrop[znear & alive5].float().mean()), 4),
                 'true_hit_rate_given_beside': round(float(drop[true_beside].float().mean()), 4)}
            for d_ in (8, 16):
                m = (n1[:, 0] == 1) & alive0[:, d_ - 1] & (n1[:, d_] >= 1)
                r[f'loc{d_}_any_above_0.3'] = round(float((peak[m, d_ - 1] > 0.3).float().mean()), 4)
                r[f'loc{d_}_peak'] = round(float(peak[m, d_ - 1].mean()), 4)
                r[f'census{d_}'] = round(float((zs[alive0[:, d_ - 1], d_ - 1] > 0.3).flatten(1).sum(1).float().mean()), 4)
            dmg.append(r)
        res['damage'] = {'per_sample': dmg, 'mean': {k: round(sum(x[k] for x in dmg) / J, 4) for k in dmg[0]}}
        # manifold
        man = {}
        for k in DEPTHS:
            m = alive0[:, k - 1]
            gk = torch.stack([v['gen'][m, k - 1].float() for v in vs]).mean(0)
            dg = SD.nearest(codes, gk, torch.device('cpu')); dt = SD.nearest(codes, fut[m, k - 1].float(), torch.device('cpu'))
            man[str(k)] = {c: [round(float(dg[cls[m, k - 1] == ci].mean()), 3), round(float(dt[cls[m, k - 1] == ci].mean()), 3)] for ci, c in enumerate(SD.CLASSES)}
        res['manifold_imagined_vs_true'] = man
        d = OUT / label
        d.mkdir(exist_ok=True)
        (d / 'scoreboard.json').write_text(json.dumps(res, indent=1))
        torch.save(rows, d / 'rows.pt')
        print(json.dumps({'scored': label, 'gen16': res['errors']['gen_over_V'][-1], 'tf1': res['errors']['tf_over_V'][0],
                          'es16_all_classes': {c: res['decomposition']['16'][c]['es'] for c in SD.CLASSES},
                          'damage': res['damage']['mean']}), flush=True)

    for v in variants:
        score([v], v)
    for gname, members in groups.items():
        score(members, gname)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
