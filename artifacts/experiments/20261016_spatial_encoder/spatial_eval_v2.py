"""Tier A, CORRECTED (SPATIAL_ENCODER_PREREG.md Amendment 4; audit ../20261016_spatial_encoder_audit/AUDIT.md). Evaluation only;
exact simulator state labels the frames (diagnosis_futures_v1), never training. The original spatial_eval.py is kept unchanged.

Corrections vs spatial_eval.py:
  split       by EPISODE SEED (Amendment 4b): the first 40 seeds of a seeded permutation (20261080) are test episodes, <= 3 roots
              each (seed 20261083); 280 fit roots are drawn (seed 20261081) from the other seeds' roots. Saved to split_v2.json.
  stages      every encoder at 'backbone_ln' (LN of last-layer patch tokens = Raw's interface) and at 'deployed' (Raw: the same;
              Spatial: patch projector). References: 'rgb' (the 147 pixel values of each 7x7 patch / 255) and an untrained ViT.
  A1          tile micro accuracy + per-class test support / recall + macro recall over present classes; ridge sensitivity
              (lambda 1e-3, 1e-2, 1e-1; fit only); zombie / cow tie-aware ROC AUC, average precision, precision / recall at the
              fit-only rate-matched threshold (strict >); HUD R^2 of the 4 stats.
  A2          persistence = same tile AND same mob layer, neither source nor destination the player cell (31); between-content
              distance from ONE seeded, label-defined set of cell pairs shared by all encoders; HUD salience over ALL displayed
              values (4 stats + inventory); counts reported. K-means K 1024, seeded init on CPU.
Usage: spatial_eval_v2.py <name>=<spec> ...   spec: raw@<step> | spatial:<ckpt> | rgb | untrained
  -> artifacts/eda/spatial_encoder_v1/tierA2_<name>.json, probes2_<name>.pt (deployed stage, Tier B readers), split_v2.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spatial_eval as EV                                                        # noqa: E402  (ridge, apply, assign, kmeans, PatchState)

import torch                                                                     # noqa: E402
from torch.nn import functional as F                                             # noqa: E402

import objective_audit as OA                                                     # noqa: E402

OUT = EV.OUT
DEV = os.environ.get('EVAL_DEVICE', 'cpu')
SPLIT = OUT / 'split_v2.json'


def episode_split():
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    seeds = [int(r['seed']) for r in rows]
    uniq = sorted(set(seeds))
    by = {s: [i for i, x in enumerate(seeds) if x == s] for s in uniq}
    perm = torch.randperm(len(uniq), generator=torch.Generator().manual_seed(20261080)).tolist()
    test_seeds, test = [], []
    g = torch.Generator().manual_seed(20261083)
    for i in perm[:40]:                                                            # Amendment 4b: 40 test episodes, <= 3 roots each
        test_seeds.append(uniq[i])
        own = by[uniq[i]]
        test += [own[j] for j in torch.randperm(len(own), generator=g)[:3].tolist()]
    pool = [i for i, x in enumerate(seeds) if x not in set(test_seeds)]
    fit = [pool[i] for i in torch.randperm(len(pool), generator=torch.Generator().manual_seed(20261081))[:280].tolist()]
    split = {'fit': fit, 'test': sorted(test), 'test_seeds': sorted(test_seeds),
             'fit_seeds': sorted({seeds[i] for i in fit}), 'shared_seeds': len({seeds[i] for i in fit} & set(test_seeds))}
    assert split['shared_seeds'] == 0
    return rows, seeds, split


def auc_tie(score, label):
    s, l = score.double(), label.bool()
    n1, n0 = int(l.sum()), int((~l).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = s.argsort(); ss = s[order]
    r = torch.empty(len(s), dtype=torch.float64)
    i = 0
    vals, counts = torch.unique_consecutive(ss, return_counts=True)
    start = 0
    for c in counts.tolist():                                                     # average ranks over ties
        r[order[start:start + c]] = (start + 1 + start + c) / 2
        start += c
    return float((r[l].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(score, label):
    s, l = score.double(), label.bool()
    if not bool(l.any()):
        return None
    o = s.argsort(descending=True)
    lt = l[o].double()
    tp = lt.cumsum(0)
    prec = tp / torch.arange(1, len(lt) + 1, dtype=torch.float64)
    return float((prec * lt).sum() / lt.sum())


class Stages:
    """frames uint8 [n,63,63,3] -> {stage: [n,81,D] float16 cpu}"""
    def __init__(self, spec):
        self.spec = spec
        if spec == 'rgb':
            self.enc = None
            return
        if spec == 'untrained':
            from d4mj.config import load_recipe
            from d4mj.lewm import LeWMEncoder
            torch.manual_seed(20261082)
            self.enc = LeWMEncoder(load_recipe(EV.ROOT / 'd4mj/recipes/lewm_mamba_raw.json')).to(DEV).eval()
            self.proj = None
            return
        os.environ['EVAL_DEVICE'] = DEV
        EV.DEV = DEV
        ps = EV.PatchState(spec)
        self.enc, self.proj = ps.enc, ps.proj

    @torch.no_grad()
    def __call__(self, frames):
        if self.enc is None:
            p = frames.float().div(255).view(-1, 9, 7, 9, 7, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 81, 147)
            return {'deployed': p.half()}
        bl, dp = [], []
        for i in range(0, len(frames), 256):
            _, _, tok, _, _ = self.enc._hidden(frames[i:i + 256].to(DEV)[:, None])
            tok = tok.float()
            bl.append(F.layer_norm(tok, (tok.shape[-1],)).half().cpu())
            if self.proj is not None:
                dp.append(self.proj(tok.reshape(-1, tok.shape[-1])).view(tok.shape).float().half().cpu())
        out = {'backbone_ln': torch.cat(bl)}
        out['deployed'] = torch.cat(dp) if dp else out['backbone_ln']
        return out


def gather(rows, idx, stages):
    """features per stage, labels: tiles [n,63], mob layer [n,63], shown [n,16], seq (n, light, sleep) per future"""
    feats = {}
    T, L, H, seq, rid = [], [], [], [], []
    for ri in idx:
        r = rows[ri]
        for f in range(5):
            alive = int((~r['future_dead'][f].bool()).long().cumprod(0).sum())
            frames = torch.cat([r['context'][-1:], r['future_frames'][f][:alive]])
            vis = torch.cat([r['root_visible'][None], r['future_visible'][f][:alive]])
            st = stages(frames)
            for k, v in st.items():
                feats.setdefault(k, []).append(v)
            t, l, sh = OA.layout(vis)
            T.append(t); L.append(l); H.append(sh); seq.append((len(frames), vis[:, 1521].float(), vis[:, 1520].float())); rid.append(ri)
    return feats, T, L, H, seq, rid


def a1(Sf, St, Tf, Tt, Lf, Lt, Hf, Ht, lam_list=(1e-3, 1e-2, 1e-1)):
    res, probes = {}, {}
    D = Sf.shape[-1]
    sub = torch.randperm(len(Sf) * 63, generator=torch.Generator().manual_seed(1))[:400000]
    Xf = Sf[:, :63].reshape(-1, D)[sub].float(); Xt = St[:, :63].reshape(-1, D)
    yf, yt = Tf.reshape(-1)[sub], Tt.reshape(-1)
    acc_l = {}
    for lam in lam_list:
        W = EV.ridge(Xf, F.one_hot(yf, 17).float(), lam=lam)
        pred = torch.cat([EV.apply(W, Xt[i:i + 200000]).argmax(-1) for i in range(0, len(Xt), 200000)])
        acc_l[str(lam)] = round(float((pred == yt).float().mean()), 4)
        if lam == 1e-2:
            probes['tile'] = W
            sup = torch.bincount(yt, minlength=17); hit = torch.bincount(yt[pred == yt], minlength=17)
            present = sup > 0
            res['tile_acc'] = acc_l[str(lam)]
            res['tile_macro_recall_present'] = round(float((hit[present].float() / sup[present].float()).mean()), 4)
            res['tile_class_support_recall'] = {str(c): [int(sup[c]), round(float(hit[c] / sup[c]), 4)] for c in range(17) if sup[c] > 0}
    res['tile_acc_by_lambda'] = acc_l
    for bit, nm in ((1, 'zombie'), (2, 'cow')):
        lf = ((Lf.reshape(-1)[sub] & bit) > 0).float(); lt = (Lt.reshape(-1) & bit) > 0
        Wm = EV.ridge(Xf, lf[:, None]); probes[nm] = Wm
        sf = EV.apply(Wm, Xf)[:, 0]
        srt = sf.sort(descending=True).values
        for mult in (0.5, 1.0, 2.0):                                              # fit-only rate-matched thresholds
            k = max(int(round(float(lf.sum()) * mult)) - 1, 0)
            probes[f'{nm}_threshold' + ('' if mult == 1.0 else f'_x{mult}')] = float(srt[min(k, len(srt) - 1)])
        st = torch.cat([EV.apply(Wm, Xt[i:i + 200000])[:, 0] for i in range(0, len(Xt), 200000)])
        th = probes[f'{nm}_threshold']
        pp = st > th
        res[nm] = {'test_positives': int(lt.sum()), 'test_cells': len(lt), 'roc_auc_tie_aware': round(auc_tie(st, lt), 4),
                   'average_precision': round(average_precision(st, lt), 4),
                   'precision_at_threshold': round(float((pp & lt).sum() / pp.sum().clamp_min(1)), 4),
                   'recall_at_threshold': round(float((pp & lt).sum() / lt.sum().clamp_min(1)), 4)}
    Wh = EV.ridge(Sf[:, 63:].reshape(len(Sf), -1).float(), Hf[:, :4] / 9); probes['hud'] = Wh
    ph = EV.apply(Wh, St[:, 63:].reshape(len(St), -1)); ht = (Ht[:, :4] / 9).float()
    r2 = 1 - (ph - ht).pow(2).mean(0) / ht.var(0).clamp_min(1e-9)
    res['hud_r2'] = {k: round(float(v), 4) for k, v in zip(('health', 'food', 'drink', 'energy'), r2)}
    e = Xt[torch.randperm(len(Xt), generator=torch.Generator().manual_seed(2))[:20000]].float()
    sv = torch.linalg.svdvals(e - e.mean(0)); p = sv / sv.sum()
    res['eff_rank'] = round(float(torch.exp(-(p * torch.log(p + 1e-12)).sum())), 2)
    res['mean_dim_std'] = round(float(e.std(0).mean()), 4)
    return res, probes, Xf, sub


def a2(Sf, St, sub, test_feats_list, T_list, L_list, H_list, seq_list, rid_list):
    D = Sf.shape[-1]
    C = EV.kmeans(torch.cat([Sf[:, :63].reshape(-1, D)[sub[:200000]], Sf[:, 63:].reshape(-1, D)[:50000]]).float(), seed=0)
    drift = {k: [0., 0] for k in ('day_steady', 'day_changing', 'night')}
    flips = {k: [0, 0] for k in ('persist', 'hud_value_changed', 'hud_unchanged')}
    between, counts = [], {'persist_pairs': 0, 'excluded_mob_change': 0, 'excluded_player': 0}
    for s, tiles, lay, sh, (n, light, sleep), rid, fi in zip(test_feats_list, T_list, L_list, H_list, seq_list, rid_list,
                                                            [i for i in range(len(rid_list))]):
        s = s.float()
        codes = EV.assign(s.reshape(-1, D), C).view(n, 81)
        g = torch.Generator().manual_seed(rid * 10 + fi % 5)                     # label-defined pairs, identical for every encoder
        for j in range(n - 1):
            dr, dc = OA.shift_of(tiles[j], tiles[j + 1]); src = OA.sources(dr, dc)
            ok = src >= 0
            tile_same = torch.zeros(63, dtype=torch.bool); tile_same[ok] = tiles[j + 1][ok] == tiles[j][src[ok]]
            mob_same = torch.zeros(63, dtype=torch.bool); mob_same[ok] = lay[j + 1][ok] == lay[j][src[ok]]
            player = torch.zeros(63, dtype=torch.bool); player[31] = True
            player[ok] |= src[ok] == 31
            same = tile_same & mob_same & ~player
            counts['excluded_mob_change'] += int((tile_same & ~mob_same & ~player).sum())
            counts['excluded_player'] += int((tile_same & mob_same & player).sum())
            idx = torch.nonzero(same)[:, 0]
            counts['persist_pairs'] += len(idx)
            if bool(sleep[j] > 0) or bool(sleep[j + 1] > 0):
                reg = None
            elif min(float(light[j]), float(light[j + 1])) < 0.5:
                reg = 'night'
            elif min(float(light[j]), float(light[j + 1])) >= 0.99 and float(light[j]) == float(light[j + 1]):
                reg = 'day_steady'
            else:
                reg = 'day_changing'
            if len(idx):
                d = (s[j + 1, idx] - s[j, src[idx]]).norm(dim=-1)
                if reg is not None:
                    drift[reg][0] += float(d.sum()); drift[reg][1] += len(idx)
                fl = codes[j + 1, idx] != codes[j, src[idx]]
                flips['persist'][0] += int(fl.sum()); flips['persist'][1] += len(idx)
            hk = 'hud_value_changed' if bool((sh[j + 1] != sh[j]).any()) else 'hud_unchanged'
            fl = codes[j + 1, 63:] != codes[j, 63:]
            flips[hk][0] += int(fl.sum()); flips[hk][1] += 18
            if j % 4 == 0:
                a_ = torch.randint(63, (64,), generator=g); b_ = torch.randint(63, (64,), generator=g)
                keep = (tiles[j][a_] != tiles[j][b_]) & (a_ != 31) & (b_ != 31)
                if bool(keep.any()):
                    between.append((s[j, a_[keep]] - s[j, b_[keep]]).norm(dim=-1))
    bd = float(torch.cat(between).mean())
    res = {'between_content_distance': round(bd, 4), 'pairs': counts,
           'drift_ratio': {k: round(v[0] / max(v[1], 1) / bd, 4) for k, v in drift.items()},
           'drift_pairs': {k: v[1] for k, v in drift.items()},
           'code_flip': {k: round(v[0] / max(v[1], 1), 4) for k, v in flips.items()},
           'code_flip_counts': {k: v[1] for k, v in flips.items()}}
    res['hud_salience'] = round(res['code_flip']['hud_value_changed'] - res['code_flip']['hud_unchanged'], 4)
    return res


@torch.no_grad()
def main():
    rows, seeds, split = episode_split()
    SPLIT.write_text(json.dumps(split))
    print(json.dumps({'fit_roots': len(split['fit']), 'test_roots': len(split['test']), 'fit_seeds': len(split['fit_seeds']),
                      'test_seeds': len(split['test_seeds']), 'shared_seeds': split['shared_seeds']}), flush=True)
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        st = Stages(spec)
        ff, Tf, Lf, Hf, _, _ = gather(rows, split['fit'], st)
        ft, Tt, Lt, Ht, seqt, ridt = gather(rows, split['test'], st)
        res = {'encoder': spec, 'split': 'episode-disjoint (split_v2.json)', 'frames_fit': sum(len(x) for x in Tf), 'frames_test': sum(len(x) for x in Tt)}
        for stage in ff:
            Sf, St = torch.cat(ff[stage]), torch.cat(ft[stage])
            r1, probes, Xf, sub = a1(Sf, St, torch.cat(Tf), torch.cat(Tt), torch.cat(Lf), torch.cat(Lt), torch.cat(Hf), torch.cat(Ht))
            r2 = a2(Sf, St, sub, ft[stage], Tt, Lt, Ht, seqt, ridt)
            res[stage] = {**r1, **r2}
            if stage == 'deployed':
                torch.save({'spec': spec, 'stage': stage, **probes}, OUT / f'probes2_{name}.pt')
            print(json.dumps({name: {stage: {k: res[stage][k] for k in ('tile_acc', 'tile_macro_recall_present', 'hud_r2', 'eff_rank', 'drift_ratio', 'code_flip', 'hud_salience')}}}), flush=True)
        (OUT / f'tierA2_{name}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
